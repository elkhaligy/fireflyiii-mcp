import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { z } from 'zod';
import { type FireflyClient, FireflyError, formatError } from '../client.js';
import {
  type JsonApiListResponse,
  type JsonApiSingleResponse,
  type UnwrappedList,
  type UnwrappedSingle,
  unwrapList,
  unwrapSingle,
} from '../transform.js';
import type { QueryParams } from '../types.js';
import { DELETE_ANNOTATIONS, READ_ANNOTATIONS, UPDATE_ANNOTATIONS, WRITE_ANNOTATIONS } from './_annotations.js';
import {
  buildTransactionListQuery,
  CATEGORY_NAME_HINT,
  dateOrDateTimeSchema,
  dateSchema,
  defineTool,
  FOREIGN_AMOUNT_HINT,
  FOREIGN_CURRENCY_CODE_HINT,
  idSchema,
  parseId,
} from './_helpers.js';
import { fetchAccountTransactions } from './accounts.js';

// A transaction response carries two ids: the top-level group `id`, which update_transaction and
// delete_transaction expect, and a `transaction_journal_id` inside each item of `transactions[]`.
// They are usually adjacent numbers, so the wrong one addresses a real but unrelated transaction
// rather than erroring — nothing at write time can tell the two apart, which is why these fields
// warn instead of validating. GROUP_ID_HINT and groupIdField both restate this fact for different
// audiences (a tool description vs. a field description); they share this one fragment so the
// wording can't drift apart.
const GROUP_VS_JOURNAL_ID =
  'the top-level `id` (the transaction group), not the `transaction_journal_id` inside `transactions[]`';

const GROUP_ID_HINT = `update_transaction and delete_transaction take ${GROUP_VS_JOURNAL_ID}.`;

const groupIdField = (verb: string): string =>
  `Transaction group ID — ${GROUP_VS_JOURNAL_ID}. That is usually an adjacent number, so the wrong one silently ${verb} a different transaction.`;

const SOURCE_ID_HINT =
  'Source account ID: an asset account or liability for withdrawals and transfers, a revenue account or liability for deposits.';
const DESTINATION_ID_HINT =
  'Destination account ID: an expense account or liability for withdrawals (e.g. a person who owes you, so their balance goes up), an asset account for deposits and transfers. A withdrawal without one lands on Firefly III\'s nameless "(no name)" expense account.';

// Optional fields shared by every transaction write tool, at the top level or per split.
const splitExtraFields = {
  bill_id: z
    .string()
    .optional()
    .describe(
      'Subscription (bill) ID to link this transaction to, so Firefly III shows the subscription as paid. Use get_bills to find valid IDs.',
    ),
  external_id: z
    .string()
    .min(1)
    .max(255)
    .optional()
    .describe('ID of this transaction in another system. Searchable with `external_id_is:<value>`.'),
  internal_reference: z
    .string()
    .min(1)
    .max(255)
    .optional()
    .describe(
      'Your own reference, e.g. the bank reference number from an SMS. Search for `internal_reference_is:<value>` with search_transactions before creating, to avoid logging the same charge twice.',
    ),
  external_url: z
    .string()
    .min(1)
    .max(255)
    .optional()
    .describe('Link to this transaction elsewhere (http or https URL).'),
};

const writeOptionFields = {
  apply_rules: z.boolean().optional().describe("Run Firefly III's rules on the result (default true)"),
  fire_webhooks: z.boolean().optional().describe('Fire Firefly III webhooks for this change (default true)'),
};

const errorIfDuplicateField = z
  .boolean()
  .optional()
  .describe(
    'When true, Firefly III refuses a transaction identical to an existing one, and the tool returns `{ "duplicate": true, "created": false, "existing_transaction_id": … }` instead of creating it. Default false. Use it when replaying bank messages that might already be logged.',
  );

// Per-split fields update_transaction accepts both flat (for one split) and inside each `splits` entry.
const updatableSplitFields = {
  source_id: z.string().optional().describe(SOURCE_ID_HINT),
  destination_id: z.string().optional().describe(DESTINATION_ID_HINT),
  category_name: z.string().optional().describe(`Category name. ${CATEGORY_NAME_HINT}`),
  budget_id: z.string().optional().describe('Budget ID'),
  currency_code: z.string().optional().describe('Currency code (e.g. EUR, USD)'),
  foreign_amount: z
    .string()
    .nullable()
    .optional()
    .describe(
      `${FOREIGN_AMOUNT_HINT} Firefly III enforces this only when creating, so an update missing it is accepted silently: when changing \`amount\` on a transaction that already has a foreign amount, send the new foreign_amount too. On its own it changes the foreign amount of a split that already has a foreign currency. Pass null to remove the foreign amount and its currency.`,
    ),
  foreign_currency_code: z
    .string()
    .optional()
    .describe(`${FOREIGN_CURRENCY_CODE_HINT} On update, always send foreign_amount with it, even if unchanged.`),
  notes: z.string().optional().describe('Additional notes'),
  tags: z.array(z.string()).optional().describe('Tags (replaces existing tags)'),
  ...splitExtraFields,
  bill_id: z
    .string()
    .nullable()
    .optional()
    .describe('Subscription (bill) ID to link this split to (use get_bills to find IDs). Pass null to unlink it.'),
};

export async function fetchTransactions(
  client: FireflyClient,
  params: {
    type?: string;
    accountId?: string;
    start?: string;
    end?: string;
    page?: number;
    limit?: number;
  },
): Promise<UnwrappedList> {
  // Firefly III's /transactions endpoint has no account_id filter and silently ignores it, so
  // accountId must be routed to /accounts/{id}/transactions instead — see issue #101.
  if (params.accountId) {
    return fetchAccountTransactions(client, parseId(params.accountId), {
      start: params.start,
      end: params.end,
      type: params.type,
      page: params.page,
      limit: params.limit,
    });
  }
  const query = buildTransactionListQuery(params);
  const response = await client.get<JsonApiListResponse>('/transactions', query);
  return unwrapList(response);
}

export async function fetchTransaction(client: FireflyClient, id: string): Promise<UnwrappedSingle> {
  const response = await client.get<JsonApiSingleResponse>(`/transactions/${id}`);
  return unwrapSingle(response);
}

type TransactionType = 'withdrawal' | 'deposit' | 'transfer';

/** Group-level flags Firefly III reads on every POST/PUT /transactions. */
interface WriteOptions {
  apply_rules?: boolean;
  fire_webhooks?: boolean;
}

/** Per-split fields that map 1:1 onto an item of `transactions[]`. */
interface SplitFields {
  source_id?: string;
  destination_id?: string;
  category_name?: string;
  budget_id?: string;
  bill_id?: string | null;
  currency_code?: string;
  foreign_amount?: string | null;
  foreign_currency_code?: string;
  notes?: string;
  tags?: string[];
  external_id?: string;
  internal_reference?: string;
  external_url?: string;
}

function writeOptions(options: WriteOptions & { error_if_duplicate_hash?: boolean }): Record<string, unknown> {
  const body: Record<string, unknown> = {
    apply_rules: options.apply_rules ?? true,
    fire_webhooks: options.fire_webhooks ?? true,
  };
  if (options.error_if_duplicate_hash !== undefined) body.error_if_duplicate_hash = options.error_if_duplicate_hash;
  return body;
}

/** What a create tool returns instead of the new transaction when Firefly III refused a duplicate. */
export interface DuplicateResult {
  duplicate: true;
  created: false;
  existing_transaction_id: string;
  message: string;
}

// With error_if_duplicate_hash, Firefly III answers 422 with "Duplicate of transaction #<group id>."
// on the first split's description. That is an expected outcome for a caller replaying the same bank
// message, so it is returned as a result rather than as a generic validation error.
const DUPLICATE_PATTERN = /Duplicate of transaction #(\d+)/;

async function postTransactionGroup(
  client: FireflyClient,
  body: Record<string, unknown>,
): Promise<UnwrappedSingle | DuplicateResult> {
  try {
    const response = await client.post<JsonApiSingleResponse>('/transactions', body);
    return unwrapSingle(response);
  } catch (err) {
    const match = err instanceof FireflyError && err.status === 422 ? DUPLICATE_PATTERN.exec(err.body) : null;
    if (!match) throw err;
    return {
      duplicate: true,
      created: false,
      existing_transaction_id: match[1],
      message: `Not created: Firefly III already has an identical transaction (group ${match[1]}).`,
    };
  }
}

const SHARED_ACCOUNTS: Record<TransactionType, Array<'source_id' | 'destination_id'>> = {
  withdrawal: ['source_id'],
  deposit: ['destination_id'],
  transfer: ['source_id', 'destination_id'],
};

const SPLIT_ACCOUNT_RULE: Record<TransactionType, string> = {
  withdrawal:
    'withdrawal splits share one source account; their destinations (expense accounts or liabilities) may differ',
  deposit: 'deposit splits share one destination account; their sources (revenue accounts or liabilities) may differ',
  transfer: 'transfer splits share both the source and the destination account',
};

/**
 * Checks Firefly III's split rules before sending anything, so a broken group fails with an
 * explanation instead of Firefly's "All accounts in this field must be equal." on one field.
 */
export function checkSplitAccounts(
  type: TransactionType,
  splits: Array<{ source_id?: string | null; destination_id?: string | null }>,
): void {
  if (splits.length < 2) return;
  for (const field of SHARED_ACCOUNTS[type]) {
    const values = new Set(splits.map((s) => (s[field] == null ? '(not set)' : String(s[field]))));
    if (values.size > 1) {
      throw new Error(
        `Firefly III rule: ${SPLIT_ACCOUNT_RULE[type]}. These splits use different ${field} values: ${[...values].join(', ')}.`,
      );
    }
  }
}

export async function createTransaction(
  client: FireflyClient,
  params: WriteOptions &
    SplitFields & {
      type: TransactionType;
      date: string;
      amount: string;
      description: string;
      error_if_duplicate_hash?: boolean;
    },
): Promise<UnwrappedSingle | DuplicateResult> {
  const { apply_rules, fire_webhooks, error_if_duplicate_hash, ...split } = params;
  return postTransactionGroup(client, {
    ...writeOptions({ apply_rules, fire_webhooks, error_if_duplicate_hash }),
    transactions: [definedFields(split)],
  });
}

/** One split (journal) of a transaction group, as returned inside `transactions[]`. */
interface GroupSplit {
  transaction_journal_id: string;
  type?: string;
  date?: string;
  description?: string;
  amount?: string;
  source_id?: string | null;
  destination_id?: string | null;
  foreign_currency_code?: string | null;
}

interface GroupSplits {
  title: string | null;
  splits: GroupSplit[];
}

async function fetchGroupSplits(client: FireflyClient, id: string): Promise<GroupSplits> {
  const group = await fetchTransaction(client, id);
  const splits = Array.isArray(group.transactions) ? (group.transactions as GroupSplit[]) : [];
  if (splits.length === 0) throw new Error(`Transaction group ${id} has no splits.`);
  return {
    title: typeof group.group_title === 'string' ? group.group_title : null,
    splits: splits.map((s) => ({ ...s, transaction_journal_id: String(s.transaction_journal_id) })),
  };
}

const describeSplits = (splits: GroupSplit[]): string =>
  splits.map((s) => `${s.transaction_journal_id} ("${s.description ?? ''}", ${s.amount ?? '?'})`).join(', ');

function definedFields(fields: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(fields).filter(([, v]) => v !== undefined));
}

interface GroupUpdatePlan extends WriteOptions {
  /** A type change, applied to every split (Firefly requires one type per group). */
  type?: unknown;
  /** Field changes per existing split, keyed by transaction_journal_id. */
  edits: Map<string, Record<string, unknown>>;
  /** New splits, sent without a transaction_journal_id. */
  additions?: Array<Record<string, unknown>>;
  /** Existing splits to leave out of the request, which makes Firefly delete them. */
  removed?: Set<string>;
  groupTitle?: string;
}

/**
 * Builds a PUT body that changes only what the plan names.
 *
 * Firefly's GroupUpdateService treats a `transactions[]` entry without `transaction_journal_id` as a
 * brand-new split, and then deletes every existing split the request did not mention (verified on
 * 6.7.9). So every split of the group that is not explicitly removed is sent with its ID; the
 * untouched ones carry nothing else, which Firefly skips. Firefly also requires all splits to share
 * one type and rejects a multi-split update without a group title.
 */
function buildGroupUpdate(group: GroupSplits, plan: GroupUpdatePlan) {
  const kept = group.splits.filter((s) => !plan.removed?.has(s.transaction_journal_id));
  const transactions: Array<Record<string, unknown>> = kept.map((s) => {
    const entry: Record<string, unknown> = { transaction_journal_id: s.transaction_journal_id };
    if (plan.type !== undefined) entry.type = plan.type;
    Object.assign(entry, plan.edits.get(s.transaction_journal_id));
    return entry;
  });
  transactions.push(...(plan.additions ?? []));
  const body: Record<string, unknown> = { ...writeOptions(plan), transactions };
  if (plan.groupTitle !== undefined) body.group_title = plan.groupTitle;
  else if (transactions.length > 1) {
    body.group_title = group.title || kept[0]?.description || group.splits[0].description || 'Split transaction';
  }
  return body;
}

/**
 * Firefly III checks foreign_amount and foreign_currency_code as a pair only when creating. On
 * update it answers 200 and stores nothing for a currency without an amount, or for an amount on
 * a split that has no foreign currency yet; an amount alone does update a split that has one.
 * It also ignores null; a foreign amount of "0" is what removes the foreign amount and currency.
 * Mutates `changes` to turn a null foreign_amount into "0".
 */
function checkForeignChange(changes: Record<string, unknown>, split: GroupSplit): void {
  if (changes.foreign_amount === null) changes.foreign_amount = '0';
  if (changes.foreign_currency_code !== undefined && changes.foreign_amount === undefined) {
    throw new Error(
      'Pass foreign_amount together with foreign_currency_code: Firefly III ignores a foreign currency sent on its own.',
    );
  }
  const clearsForeign = changes.foreign_amount !== undefined && Number(changes.foreign_amount) === 0;
  if (changes.foreign_amount !== undefined && changes.foreign_currency_code === undefined) {
    if (!clearsForeign && !split.foreign_currency_code) {
      throw new Error(
        `Split ${split.transaction_journal_id} has no foreign amount yet, so pass foreign_currency_code with foreign_amount: Firefly III ignores a foreign amount without a currency.`,
      );
    }
  }
}

/** One entry of update_transaction's `splits`: an edit when it names a journal, otherwise a new split. */
export interface SplitChange extends SplitFields {
  transaction_journal_id?: string;
  amount?: string;
  description?: string;
  date?: string;
}

const isTransactionType = (value: unknown): value is TransactionType =>
  value === 'withdrawal' || value === 'deposit' || value === 'transfer';

/** Turns update_transaction's `splits` / `remove_journal_ids` into a plan, validating against the stored group. */
function planSplitChanges(
  id: string,
  group: GroupSplits,
  splits: SplitChange[],
  removeIds: string[],
  type: TransactionType | undefined,
): Pick<GroupUpdatePlan, 'edits' | 'additions' | 'removed'> {
  const byId = new Map(group.splits.map((s) => [s.transaction_journal_id, s]));
  const notInGroup = (journalId: string) =>
    new Error(
      `Split ${journalId} is not part of transaction group ${id}. Its splits are: ${describeSplits(group.splits)}.`,
    );
  const removed = new Set(removeIds);
  for (const journalId of removed) if (!byId.has(journalId)) throw notInGroup(journalId);

  const edits = new Map<string, Record<string, unknown>>();
  const newSplits: Array<Record<string, unknown>> = [];
  for (const { transaction_journal_id: journalId, ...fields } of splits) {
    const changes = definedFields(fields);
    if (journalId === undefined) {
      if (changes.amount === undefined || changes.description === undefined) {
        throw new Error(
          'A new split (an entry of splits without transaction_journal_id) needs amount and description.',
        );
      }
      if (changes.foreign_amount === null) delete changes.foreign_amount;
      if ((changes.foreign_amount === undefined) !== (changes.foreign_currency_code === undefined)) {
        throw new Error('A new split needs foreign_amount and foreign_currency_code together, or neither.');
      }
      newSplits.push(changes);
      continue;
    }
    const split = byId.get(journalId);
    if (!split) throw notInGroup(journalId);
    if (removed.has(journalId))
      throw new Error(`Split ${journalId} is both edited in splits and listed in remove_journal_ids.`);
    if (edits.has(journalId))
      throw new Error(`Split ${journalId} appears more than once in splits; merge its changes into one entry.`);
    checkForeignChange(changes, split);
    edits.set(journalId, changes);
  }

  const kept = group.splits.filter((s) => !removed.has(s.transaction_journal_id));
  if (kept.length + newSplits.length === 0) {
    throw new Error('This would remove every split of the transaction. Use delete_transaction to delete it instead.');
  }

  // A new split inherits the group's type, date and shared account(s) unless it sets its own, so
  // adding "Mody's part" to a withdrawal only needs amount, description and destination_id.
  const template = kept[0] ?? group.splits[0];
  const finalType = type ?? (isTransactionType(template.type) ? template.type : undefined);
  const additions = newSplits.map((changes) => {
    const entry: Record<string, unknown> = {};
    if (finalType !== undefined) entry.type = finalType;
    if (template.date !== undefined) entry.date = template.date;
    for (const field of finalType ? SHARED_ACCOUNTS[finalType] : []) {
      if (template[field] != null) entry[field] = template[field];
    }
    return Object.assign(entry, changes);
  });

  if (finalType !== undefined) {
    checkSplitAccounts(finalType, [
      ...kept.map((s) => ({
        source_id: (edits.get(s.transaction_journal_id)?.source_id as string | undefined) ?? s.source_id,
        destination_id: (edits.get(s.transaction_journal_id)?.destination_id as string | undefined) ?? s.destination_id,
      })),
      ...additions.map((a) => ({
        source_id: a.source_id as string | undefined,
        destination_id: a.destination_id as string | undefined,
      })),
    ]);
  }
  return { edits, additions, removed };
}

export async function updateTransaction(
  client: FireflyClient,
  id: string,
  params: WriteOptions &
    SplitFields & {
      transaction_journal_id?: string;
      type?: TransactionType;
      date?: string;
      amount?: string;
      description?: string;
      group_title?: string;
      splits?: SplitChange[];
      remove_journal_ids?: string[];
    },
): Promise<UnwrappedSingle> {
  const {
    transaction_journal_id: journalId,
    type,
    group_title: groupTitle,
    splits,
    remove_journal_ids: removeIds,
    apply_rules,
    fire_webhooks,
    ...fields
  } = params;
  const flatChanges = definedFields(fields);
  const perSplit = splits !== undefined || removeIds !== undefined;

  if (perSplit && (journalId !== undefined || Object.keys(flatChanges).length > 0)) {
    throw new Error(
      `When using splits or remove_journal_ids, put per-split fields inside splits (one entry per split) instead of at the top level. Top-level fields that still apply: type, group_title. Got top-level: ${[...(journalId !== undefined ? ['transaction_journal_id'] : []), ...Object.keys(flatChanges)].join(', ')}.`,
    );
  }
  if (!perSplit && Object.keys(flatChanges).length === 0 && type === undefined && groupTitle === undefined) {
    throw new Error('Nothing to update: pass at least one field to change.');
  }
  if (perSplit && (splits ?? []).length === 0 && (removeIds ?? []).length === 0) {
    throw new Error('Nothing to update: splits and remove_journal_ids are both empty.');
  }

  const group = await fetchGroupSplits(client, id);
  const plan: GroupUpdatePlan = { type, groupTitle, apply_rules, fire_webhooks, edits: new Map() };

  if (perSplit) {
    Object.assign(plan, planSplitChanges(id, group, splits ?? [], removeIds ?? [], type));
  } else if (Object.keys(flatChanges).length > 0) {
    let target: string;
    if (journalId !== undefined) {
      if (!group.splits.some((s) => s.transaction_journal_id === journalId)) {
        throw new Error(
          `Split ${journalId} is not part of transaction group ${id}. Its splits are: ${describeSplits(group.splits)}.`,
        );
      }
      target = journalId;
    } else if (group.splits.length === 1) {
      target = group.splits[0].transaction_journal_id;
    } else {
      throw new Error(
        `Transaction group ${id} has ${group.splits.length} splits; pass transaction_journal_id to choose which one to update, or use splits to change several at once. Splits: ${describeSplits(group.splits)}.`,
      );
    }
    const split = group.splits.find((s) => s.transaction_journal_id === target) as GroupSplit;
    checkForeignChange(flatChanges, split);
    plan.edits.set(target, flatChanges);
  }

  const response = await client.put<JsonApiSingleResponse>(`/transactions/${id}`, buildGroupUpdate(group, plan));
  return unwrapSingle(response);
}

export async function deleteTransaction(client: FireflyClient, id: string): Promise<{ deleted: true; id: string }> {
  await client.delete(`/transactions/${id}`);
  return { deleted: true, id };
}

export async function searchTransactions(
  client: FireflyClient,
  params: { query: string; page?: number; limit?: number },
): Promise<UnwrappedList> {
  const query: QueryParams = { query: params.query, page: params.page, limit: params.limit };
  const response = await client.get<JsonApiListResponse>('/search/transactions', query);
  return unwrapList(response);
}

export const BULK_UPDATE_DEFAULT_MAX = 50;
export const BULK_UPDATE_HARD_MAX = 500;
const BULK_SEARCH_PAGE_SIZE = 100;

export interface BulkUpdateResult {
  matched: number;
  updated: Array<{ id: string; splits: string[] }>;
  failed: Array<{ id: string; error: string }>;
}

/**
 * Applies the same field changes to every transaction a search query matches.
 *
 * Firefly's own `/data/bulk/transactions` endpoint can only move transactions between accounts
 * (`where.account_id` → `update.account_id`), so this is done client-side: page through the search
 * results, then update each matched group. Search returns *partial* groups (only the splits that
 * matched), and sending those back as-is would make Firefly delete the unmatched splits, so each
 * group is re-read in full and only the matched splits are changed (see buildGroupUpdate).
 *
 * Nothing is changed if the query matches more than `max_transactions` groups.
 */
export async function bulkUpdateTransactions(
  client: FireflyClient,
  params: {
    query: string;
    category_name?: string;
    budget_id?: string;
    tags?: string[];
    notes?: string;
    max_transactions?: number;
  },
): Promise<BulkUpdateResult> {
  const changes = definedFields({
    category_name: params.category_name,
    budget_id: params.budget_id,
    tags: params.tags,
    notes: params.notes,
  });
  if (Object.keys(changes).length === 0) {
    throw new Error('Nothing to update: pass at least one of category_name, budget_id, tags or notes.');
  }
  const max = params.max_transactions ?? BULK_UPDATE_DEFAULT_MAX;

  const matched = new Map<string, Set<string>>();
  for (let page = 1; ; page++) {
    const result = await searchTransactions(client, { query: params.query, page, limit: BULK_SEARCH_PAGE_SIZE });
    for (const group of result.data) {
      const splits = matched.get(group.id) ?? new Set<string>();
      for (const split of (group.transactions as GroupSplit[] | undefined) ?? []) {
        splits.add(String(split.transaction_journal_id));
      }
      matched.set(group.id, splits);
    }
    if (matched.size > max) {
      throw new Error(
        `The query matches more than ${max} transactions, so nothing was changed. Narrow the query, or raise max_transactions (up to ${BULK_UPDATE_HARD_MAX}) after checking the matches with search_transactions.`,
      );
    }
    if (result.data.length === 0 || !result.pagination || page >= result.pagination.totalPages) break;
  }

  const outcome: BulkUpdateResult = { matched: matched.size, updated: [], failed: [] };
  for (const [id, splitIds] of matched) {
    try {
      const group = await fetchGroupSplits(client, id);
      await client.put(
        `/transactions/${id}`,
        buildGroupUpdate(group, { edits: new Map([...splitIds].map((journalId) => [journalId, changes])) }),
      );
      outcome.updated.push({ id, splits: [...splitIds] });
    } catch (err) {
      outcome.failed.push({ id, error: formatError(err) });
    }
  }
  return outcome;
}

export async function createSplitTransaction(
  client: FireflyClient,
  params: WriteOptions & {
    type: TransactionType;
    date: string;
    source_id?: string;
    destination_id?: string;
    currency_code?: string;
    foreign_currency_code?: string;
    group_title?: string;
    error_if_duplicate_hash?: boolean;
    splits: Array<
      Omit<SplitFields, 'currency_code' | 'foreign_currency_code' | 'foreign_amount'> & {
        amount: string;
        foreign_amount?: string;
        description: string;
      }
    >;
  },
): Promise<UnwrappedSingle | DuplicateResult> {
  // Top-level accounts are defaults; a split that sets its own source_id/destination_id wins.
  const transactions = params.splits.map((split) =>
    definedFields({
      type: params.type,
      date: params.date,
      currency_code: params.currency_code,
      foreign_currency_code: params.foreign_currency_code,
      ...split,
      source_id: split.source_id ?? params.source_id,
      destination_id: split.destination_id ?? params.destination_id,
    }),
  );
  checkSplitAccounts(params.type, transactions as Array<{ source_id?: string; destination_id?: string }>);
  return postTransactionGroup(client, {
    ...writeOptions(params),
    group_title: params.group_title ?? params.splits[0].description,
    transactions,
  });
}

export function registerTransactionTools(server: McpServer, client: FireflyClient): void {
  defineTool(
    server,
    'get_transactions',
    {
      title: 'Get Transactions',
      description:
        'Get transactions from Firefly III. Filter by type (withdrawal/deposit/transfer/reconciliation), account ID, or date range. Dates must be YYYY-MM-DD. Use get_transaction to fetch a single transaction by ID.',
      inputSchema: {
        type: z
          .enum(['withdrawal', 'deposit', 'transfer', 'reconciliation'])
          .optional()
          .describe('Transaction type filter'),
        accountId: z
          .string()
          .optional()
          .describe(
            'Filter by account ID — use get_accounts to find valid IDs. Internally delegates to the same endpoint as get_account_transactions.',
          ),
        start: dateSchema.optional().describe('Start date (YYYY-MM-DD)'),
        end: dateSchema.optional().describe('End date (YYYY-MM-DD)'),
        page: z.number().int().positive().optional().default(1).describe('Page number'),
        limit: z.number().int().positive().max(100).optional().default(50).describe('Results per page (max 100)'),
      },
      annotations: READ_ANNOTATIONS,
    },
    ({ type, accountId, start, end, page, limit }) =>
      fetchTransactions(client, {
        type: type,
        accountId: accountId,
        start: start,
        end: end,
        page: page,
        limit: limit,
      }),
  );

  defineTool(
    server,
    'get_transaction',
    {
      title: 'Get Transaction',
      description: `Get a single Firefly III transaction by its numeric ID, including all splits. Use get_transactions to find valid transaction IDs. ${GROUP_ID_HINT}`,
      inputSchema: {
        id: idSchema.describe('Transaction ID'),
      },
      annotations: READ_ANNOTATIONS,
    },
    ({ id }) => fetchTransaction(client, id),
  );

  defineTool(
    server,
    'create_transaction',
    {
      title: 'Create Transaction',
      description: `Create a new transaction in Firefly III. Use get_accounts to find source and destination account IDs and their currencies; a transfer between accounts in different currencies also needs foreign_amount and foreign_currency_code. For a card charge billed in EGP but made in USD, pass amount "912.36" plus foreign_amount "16.99" and foreign_currency_code "USD". Set bill_id to mark a subscription as paid, and internal_reference + error_if_duplicate_hash: true when logging from bank messages. ${GROUP_ID_HINT}`,
      inputSchema: {
        type: z.enum(['withdrawal', 'deposit', 'transfer']).describe('Transaction type'),
        date: dateOrDateTimeSchema.describe('Transaction date (YYYY-MM-DD or RFC 3339 date-time with timezone)'),
        amount: z.string().describe('Amount as a positive number string, e.g. "42.50"'),
        description: z.string().describe('Short description of the transaction'),
        source_id: z.string().optional().describe(`${SOURCE_ID_HINT} Required for withdrawals and transfers.`),
        destination_id: z.string().optional().describe(`${DESTINATION_ID_HINT} Required for deposits and transfers.`),
        category_name: z.string().optional().describe(`Category name to assign. ${CATEGORY_NAME_HINT}`),
        budget_id: z.string().optional().describe('Budget ID — use get_budgets to find valid IDs'),
        currency_code: z.string().optional().describe('Currency code (e.g. EUR, USD). Defaults to account currency.'),
        foreign_amount: z.string().optional().describe(FOREIGN_AMOUNT_HINT),
        foreign_currency_code: z.string().optional().describe(FOREIGN_CURRENCY_CODE_HINT),
        notes: z.string().optional().describe('Additional notes'),
        tags: z.array(z.string()).optional().describe('Tags to attach'),
        ...splitExtraFields,
        error_if_duplicate_hash: errorIfDuplicateField,
        ...writeOptionFields,
      },
      annotations: WRITE_ANNOTATIONS,
    },
    (params) => createTransaction(client, params),
  );

  defineTool(
    server,
    'update_transaction',
    {
      title: 'Update Transaction',
      description: `Update an existing transaction in Firefly III. Only fields provided are changed; splits you don't mention are always kept as they are. Two ways to call it:
- Flat fields (amount, category_name, ...) change one split: the only one, or the one named by transaction_journal_id.
- \`splits\` changes several splits in one call and can add new ones: an entry with transaction_journal_id edits that split, an entry without one adds a split (it inherits the group's type, date and shared account). Add remove_journal_ids to delete specific splits.
Example — turn a 2000 EGP card charge into "1000 mine + 1000 for Mody": {"id": "57", "group_title": "Amazon order", "splits": [{"transaction_journal_id": "221", "amount": "1000.00"}, {"amount": "1000.00", "description": "Mody's part", "destination_id": "12"}]}.
type and group_title apply to the whole group. Use get_transaction to confirm the IDs before updating. ${GROUP_ID_HINT}`,
      inputSchema: {
        id: idSchema.describe(groupIdField('updates')),
        transaction_journal_id: z
          .string()
          .optional()
          .describe(
            "Which split the flat fields update: the `transaction_journal_id` from the group's `transactions[]`. Optional when the transaction has a single split; required when it has several (or use splits).",
          ),
        type: z
          .enum(['withdrawal', 'deposit', 'transfer'])
          .optional()
          .describe('Transaction type, applied to every split (Firefly III requires one type per group)'),
        group_title: z.string().min(1).optional().describe('Title for the whole transaction group'),
        date: dateOrDateTimeSchema
          .optional()
          .describe('Transaction date (YYYY-MM-DD or RFC 3339 date-time with timezone)'),
        amount: z.string().optional().describe('Amount as a positive number string'),
        description: z.string().optional().describe('Short description'),
        ...updatableSplitFields,
        splits: z
          .array(
            z.object({
              transaction_journal_id: z
                .string()
                .optional()
                .describe(
                  'The split to edit. Leave it out to add a new split (which then needs amount and description).',
                ),
              amount: z.string().optional().describe('Amount as a positive number string'),
              description: z.string().optional().describe('Description of this split'),
              date: dateOrDateTimeSchema.optional().describe('Date of this split (YYYY-MM-DD or RFC 3339 date-time)'),
              ...updatableSplitFields,
            }),
          )
          .optional()
          .describe(
            'Per-split changes; cannot be combined with the flat per-split fields. Splits not listed here are kept unchanged.',
          ),
        remove_journal_ids: z
          .array(z.string())
          .optional()
          .describe(
            'transaction_journal_ids of splits to delete from the group. Nothing is ever removed unless listed here.',
          ),
        ...writeOptionFields,
      },
      annotations: UPDATE_ANNOTATIONS,
    },
    ({ id, ...params }) => updateTransaction(client, id, params),
  );

  defineTool(
    server,
    'delete_transaction',
    {
      title: 'Delete Transaction',
      description:
        'Permanently delete a transaction from Firefly III. **This action cannot be undone.** Use get_transaction to confirm the transaction before deleting.',
      inputSchema: {
        id: idSchema.describe(groupIdField('deletes')),
      },
      annotations: DELETE_ANNOTATIONS,
    },
    ({ id }) => deleteTransaction(client, id),
  );

  defineTool(
    server,
    'search_transactions',
    {
      title: 'Search Transactions',
      description:
        'Search for transactions in Firefly III by keyword. Searches across descriptions, notes, and other fields. Supports Firefly III search operators, e.g. `internal_reference_is:ABC123` or `external_id_is:sms-42` (check before creating a transaction from a bank message), `bill_is:Spotify`, `category_is:Food`, `date_after:2026-10-01`.',
      inputSchema: {
        query: z.string().describe('Search query'),
        page: z.number().int().positive().optional().default(1).describe('Page number'),
        limit: z.number().int().positive().max(100).optional().default(50).describe('Results per page (max 100)'),
      },
      annotations: READ_ANNOTATIONS,
    },
    ({ query, page, limit }) =>
      searchTransactions(client, {
        query: query,
        page: page,
        limit: limit,
      }),
  );

  defineTool(
    server,
    'create_split_transaction',
    {
      title: 'Create Split Transaction',
      description: `Create a split transaction in Firefly III: one receipt divided across several categories, budgets, descriptions or accounts. All splits share the type and date. Top-level source_id/destination_id are defaults that a split can override, following Firefly III's rules: withdrawal splits share one source but may go to different destinations (expense accounts or liabilities); deposit splits share one destination but may come from different sources; transfer splits share both. Example — a 2000 EGP card charge where half was bought for someone who owes you: {"type": "withdrawal", "date": "2026-10-23", "source_id": "3", "group_title": "Amazon order", "splits": [{"amount": "1000.00", "description": "My part", "destination_id": "41"}, {"amount": "1000.00", "description": "Mody's part", "destination_id": "12"}]}. Use get_accounts to find account IDs. ${GROUP_ID_HINT}`,
      inputSchema: {
        type: z.enum(['withdrawal', 'deposit', 'transfer']).describe('Transaction type (shared across all splits)'),
        date: dateOrDateTimeSchema.describe(
          'Transaction date (YYYY-MM-DD or RFC 3339 date-time with timezone, shared across all splits)',
        ),
        source_id: z
          .string()
          .optional()
          .describe(
            `Default source account for every split. ${SOURCE_ID_HINT} Required for withdrawals and transfers.`,
          ),
        destination_id: z
          .string()
          .optional()
          .describe(`Default destination account for every split. ${DESTINATION_ID_HINT}`),
        currency_code: z.string().optional().describe('Currency code (e.g. EUR, USD). Defaults to account currency.'),
        foreign_currency_code: z
          .string()
          .optional()
          .describe(`${FOREIGN_CURRENCY_CODE_HINT} Shared across all splits; set foreign_amount on every split.`),
        group_title: z
          .string()
          .min(1)
          .optional()
          .describe(
            "Title for the whole transaction group (Firefly III requires one when there are several splits). Defaults to the first split's description.",
          ),
        splits: z
          .array(
            z.object({
              amount: z.string().describe('Amount as a positive number string, e.g. "42.50"'),
              foreign_amount: z.string().optional().describe(`Foreign amount for this split. ${FOREIGN_AMOUNT_HINT}`),
              description: z.string().describe('Description for this split'),
              source_id: z.string().optional().describe('Source account for this split (overrides the top-level one)'),
              destination_id: z
                .string()
                .optional()
                .describe('Destination account for this split (overrides the top-level one)'),
              category_name: z.string().optional().describe(`Category name. ${CATEGORY_NAME_HINT}`),
              budget_id: z.string().optional().describe('Budget ID — use get_budgets to find valid IDs'),
              tags: z.array(z.string()).optional().describe('Tags'),
              notes: z.string().optional().describe('Notes'),
              ...splitExtraFields,
            }),
          )
          .min(2)
          .describe('At least 2 splits required'),
        error_if_duplicate_hash: errorIfDuplicateField,
        ...writeOptionFields,
      },
      annotations: WRITE_ANNOTATIONS,
    },
    (params) => createSplitTransaction(client, params),
  );

  defineTool(
    server,
    'bulk_update_transactions',
    {
      title: 'Bulk Update Transactions',
      description:
        'Set the category, budget, tags or notes on every transaction a search query matches (same syntax as search_transactions). Only the splits the query matches are changed. Run search_transactions with the same query first to check what will change: if the query matches more than max_transactions transactions, nothing is changed. Returns the updated and failed transaction IDs.',
      inputSchema: {
        query: z.string().describe('Search query to select transactions (same syntax as search_transactions)'),
        category_name: z
          .string()
          .optional()
          .describe(`Set category for all matched transactions. ${CATEGORY_NAME_HINT}`),
        budget_id: z.string().optional().describe('Set budget for all matched transactions'),
        tags: z.array(z.string()).optional().describe('Replace tags on all matched transactions'),
        notes: z.string().optional().describe('Set notes on all matched transactions'),
        max_transactions: z
          .number()
          .int()
          .positive()
          .max(BULK_UPDATE_HARD_MAX)
          .optional()
          .default(BULK_UPDATE_DEFAULT_MAX)
          .describe(
            `Safety limit: refuse to change anything if the query matches more transactions than this (default ${BULK_UPDATE_DEFAULT_MAX}, max ${BULK_UPDATE_HARD_MAX})`,
          ),
      },
      annotations: WRITE_ANNOTATIONS,
    },
    (params) => bulkUpdateTransactions(client, params),
  );
}
