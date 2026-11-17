export type AgentState = "idle" | "thinking" | "tool" | "talking";

export interface Agent {
  id: string;
  name: string;
  role: string;
  instructions: string;
  tier: string;
  model: string | null;
  tools: string[];
  parent_id: string | null;
  is_orchestrator: boolean;
  box_ids: string[];
  state: AgentState;
  detail: string;
  peer: string | null;
}

export interface RoomInfo { id: string; name: string; created_at: string; agents?: number }

export interface Doc { id: string; filename: string; n_chunks: number }
export interface Box { id: string; name: string; description: string; documents: Doc[] }

export interface Task {
  id: string;
  title: string;
  brief: string;
  status: string;
  assignee_id: string;
  created_by: string;
  parent_id: string | null;
  result: string;
  handoffs: number;
  room_id: string;
  created_at: string;
}

export interface Message {
  id: string;
  sender: string;
  recipient: string;
  kind: string;
  content: string;
  task_id: string | null;
  trace_id: string | null;
  room_id: string;
  created_at: string;
}

export interface Settings {
  has_key: boolean;
  key_hint: string;
  tiers: Record<string, string>;
  judge_enabled: boolean;
  router_mode: "heuristic" | "learned";
  explore_rate: number;
}

export interface Config { langfuse: boolean; optional_tools: string[]; embedding_model: string }
export interface Model { id: string; name: string; input: number; output: number }
export interface StatRow { key: string; runs: number; judged: number; avg_score: number | null; avg_cost: number | null; total_cost: number | null }
export interface RouterStats { by_tier: StatRow[]; by_router: StatRow[]; trained_tiers: Record<string, number>; usable: boolean; min_samples: number }
export interface TrainResult { samples: number; tiers_trained: string[]; usable: boolean }

export interface State { room: RoomInfo; agents: Agent[]; boxes: Box[]; tasks: Task[]; messages: Message[]; settings: Settings; config: Config }

export type Event =
  | { type: "state"; agent_id: string; state: AgentState; detail: string; peer: string | null }
  | { type: "message"; message: Message }
  | { type: "task"; task: Task };

/** Returns the Clerk session token, or null in local dev-auth mode. Set once at startup. */
let getToken: () => Promise<string | null> = async () => null;
export function setTokenGetter(fn: () => Promise<string | null>) { getToken = fn; }

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const form = body instanceof FormData;
  const token = await getToken();
  const res = await fetch(`/api${path}`, {
    method,
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(body && !form ? { "Content-Type": "application/json" } : {}),
    },
    body: form ? (body as FormData) : body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(typeof err.detail === "string" ? err.detail : `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  publicConfig: () => fetch("/api/public").then((r) => r.json() as Promise<{ clerk_publishable_key: string; dev_auth: boolean }>),
  state: (roomId: string | null) => req<State>("GET", roomId ? `/state?room_id=${roomId}` : "/state"),
  rooms: () => req<RoomInfo[]>("GET", "/rooms"),
  createRoom: (name: string) => req<RoomInfo>("POST", "/rooms", { name }),
  renameRoom: (id: string, name: string) => req<RoomInfo>("PATCH", `/rooms/${id}`, { name }),
  deleteRoom: (id: string) => req("DELETE", `/rooms/${id}`),
  chat: (agent_id: string, content: string) => req<Message>("POST", "/chat", { agent_id, content }),
  saveAgent: (id: string | null, data: Partial<Agent> & { room_id?: string }) =>
    id ? req<Agent>("PATCH", `/agents/${id}`, data) : req<Agent>("POST", "/agents", data),
  deleteAgent: (id: string) => req("DELETE", `/agents/${id}`),
  createBox: (name: string, description: string) => req<Box>("POST", "/boxes", { name, description }),
  deleteBox: (id: string) => req("DELETE", `/boxes/${id}`),
  upload: (boxId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req<Doc>("POST", `/boxes/${boxId}/files`, fd);
  },
  retryTask: (id: string) => req<{ restarted: number }>("POST", `/tasks/${id}/retry`),
  feedback: (messageId: string, value: number) => req("POST", `/messages/${messageId}/feedback`, { value }),
  demo: (roomId: string) => req("POST", `/demo?room_id=${roomId}`),
  saveSettings: (data: Partial<Settings> & { openrouter_key?: string } & Record<string, unknown>) => req<Settings>("PUT", "/settings", data),
  models: () => req<Model[]>("GET", "/models"),
  routerStats: () => req<RouterStats>("GET", "/router"),
  trainRouter: () => req<TrainResult>("POST", "/router/train"),
};

/** Live event stream; reconnects on drop (with a fresh token) and calls onOpen so the caller can resync. */
export function connect(onEvent: (e: Event) => void, onOpen: () => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  const open = async () => {
    const token = await getToken();
    if (closed) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/api/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`);
    ws.onopen = onOpen;
    ws.onmessage = (m) => onEvent(JSON.parse(m.data));
    ws.onclose = () => { if (!closed) setTimeout(open, 1500); };
  };
  open();
  return () => { closed = true; ws?.close(); };
}
