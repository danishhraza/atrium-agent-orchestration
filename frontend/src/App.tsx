import { useCallback, useEffect, useRef, useState, type ReactElement, type ReactNode } from "react";
import {
  api, connect, type Agent, type Box, type Config, type Event, type Message, type Model, type RoomInfo, type RouterStats,
  type Settings, type StatRow, type Task,
} from "./api";
import Markdown from "./Markdown";
import Room, { type Packet } from "./Room";

type Tab = "chat" | "tasks" | "comms" | "knowledge";
type Run = (fn: () => Promise<unknown>) => Promise<void>;
const TIERS = ["fast", "balanced", "smart"];

function upsert<T extends { id: string }>(list: T[], item: T): T[] {
  return list.some((x) => x.id === item.id) ? list.map((x) => (x.id === item.id ? item : x)) : [...list, item];
}

const ROOM_KEY = "atrium.room";
function savedRoom(): string | null {
  try { return localStorage.getItem(ROOM_KEY); } catch { return null; }
}
function rememberRoom(id: string) {
  try { localStorage.setItem(ROOM_KEY, id); } catch { /* private mode: just don't remember */ }
}

/** Switch, create, rename and delete rooms. Each room is its own team with its own Jarvis. */
function RoomMenu({ rooms, current, onSwitch, run, onError }: {
  rooms: RoomInfo[]; current: RoomInfo | null; onSwitch: (id: string) => void; run: Run; onError: (e: string) => void;
}) {
  const [mode, setMode] = useState<"" | "new" | "rename">("");
  const [name, setName] = useState("");
  const close = () => { setMode(""); setName(""); };
  const submit = async () => {
    const n = name.trim();
    if (!n) return;
    try {
      if (mode === "new") onSwitch((await api.createRoom(n)).id);
      else if (current) await run(() => api.renameRoom(current.id, n));
      close();
    } catch (e) { onError((e as Error).message); }
  };
  const remove = () => {
    if (!current) return;
    if (!confirm(`Delete "${current.name}"? Its agents, tasks and conversation are deleted. Knowledge boxes are kept.`)) return;
    api.deleteRoom(current.id)
      .then(() => onSwitch(rooms.find((r) => r.id !== current.id)!.id))
      .catch((e) => onError(e.message));
  };

  if (mode) {
    return (
      <div className="roommenu">
        <input autoFocus value={name} placeholder={mode === "new" ? "New room name" : "Rename room"} aria-label="Room name"
               onChange={(e) => setName(e.target.value)}
               onKeyDown={(e) => { if (e.key === "Enter") submit(); if (e.key === "Escape") close(); }} />
        <button className="primary" disabled={!name.trim()} onClick={submit}>{mode === "new" ? "Create" : "Save"}</button>
        <button onClick={close}>Cancel</button>
      </div>
    );
  }
  return (
    <div className="roommenu">
      <select value={current?.id ?? ""} onChange={(e) => onSwitch(e.target.value)} aria-label="Room">
        {rooms.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
      </select>
      <button onClick={() => setMode("new")} title="New room">+ Room</button>
      <button onClick={() => { setName(current?.name ?? ""); setMode("rename"); }} title="Rename room" aria-label="Rename room">✎</button>
      {rooms.length > 1 && <button className="danger" onClick={remove} title="Delete room" aria-label="Delete room">🗑</button>}
    </div>
  );
}

export default function App({ account }: { account: ReactNode }) {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [boxes, setBoxes] = useState<Box[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [tab, setTab] = useState<Tab>("chat");
  const [selected, setSelected] = useState<string | null>(null);
  const [editing, setEditing] = useState<Agent | "new" | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [bubbles, setBubbles] = useState<Record<string, string>>({});
  const [packets, setPackets] = useState<Packet[]>([]);
  const [error, setError] = useState("");
  const [rooms, setRooms] = useState<RoomInfo[]>([]);
  const [room, setRoom] = useState<RoomInfo | null>(null);
  const roomId = useRef<string | null>(savedRoom());

  const refresh = useCallback(async () => {
    try {
      let s;
      try {
        s = await api.state(roomId.current);
      } catch {  // the remembered room was deleted (or belongs to another login): fall back to the first room
        roomId.current = null;
        s = await api.state(null);
      }
      roomId.current = s.room.id;
      rememberRoom(s.room.id);
      setRoom(s.room);
      api.rooms().then(setRooms).catch(() => {});
      setAgents(s.agents); setBoxes(s.boxes); setTasks(s.tasks); setMessages(s.messages);
      setSettings(s.settings); setConfig(s.config);
      setSelected((cur) => (cur && s.agents.some((a) => a.id === cur) ? cur : s.agents.find((a) => a.is_orchestrator)?.id ?? null));
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    const onEvent = (e: Event) => {
      if (e.type === "state") {
        setAgents((list) => list.map((a) => (a.id === e.agent_id ? { ...a, state: e.state, detail: e.detail, peer: e.peer } : a)));
      } else if (e.type === "task") {
        if (e.task.room_id === roomId.current) setTasks((list) => upsert(list, e.task));
      } else {
        const m = e.message;
        if (m.room_id !== roomId.current) return;  // events from your other rooms
        const snippet = m.content.slice(0, 90);
        setMessages((list) => upsert(list, m));
        setPackets((list) => [...list, { id: m.id, from: m.sender, to: m.recipient, kind: m.kind }]);
        setTimeout(() => setPackets((list) => list.filter((p) => p.id !== m.id)), 1200);
        if (m.sender !== "user") {
          setBubbles((b) => ({ ...b, [m.sender]: snippet }));
          setTimeout(() => setBubbles((b) => (b[m.sender] === snippet ? { ...b, [m.sender]: "" } : b)), 6000);
        }
      }
    };
    return connect(onEvent, refresh);
  }, [refresh]);

  const run: Run = async (fn) => {
    try { await fn(); await refresh(); } catch (e) { setError((e as Error).message); }
  };
  const switchRoom = (id: string) => {
    roomId.current = id;
    rememberRoom(id);
    setSelected(null); setTab("chat"); setBubbles({}); setPackets([]);
    refresh();
  };
  const name = (id: string) => (id === "user" ? "You" : agents.find((a) => a.id === id)?.name ?? "former agent");
  const current = agents.find((a) => a.id === selected);
  const openTasks = tasks.filter((t) => !["done", "failed"].includes(t.status)).length;
  const needsKey = settings && !settings.has_key;

  return (
    <div className="app">
      <Room agents={agents} bubbles={bubbles} packets={packets} selected={selected} openTasks={openTasks}
            onSelect={(id) => { setSelected(id); setTab("chat"); }} onEdit={setEditing} />

      <header className="hud top">
        <div className="logo">Atrium</div>
        <RoomMenu rooms={rooms} current={room} onSwitch={switchRoom} run={run} onError={setError} />
        {config?.langfuse && <span className="pill ok">Langfuse on</span>}
        <span className="grow" />
        {agents.length === 1 && !needsKey && <button onClick={() => room && run(() => api.demo(room.id))}>Load demo team</button>}
        <button className="primary" onClick={() => setEditing("new")}>+ New agent</button>
        <button onClick={() => setShowSettings(true)} aria-label="Settings">⚙ Settings</button>
        {account}
      </header>

      {needsKey && (
        <div className="hud banner">
          <b>Your team is asleep.</b> Add your OpenRouter API key to wake them up.
          <button className="primary" onClick={() => setShowSettings(true)}>Add key</button>
        </div>
      )}
      {error && <div className="hud banner bad" onClick={() => setError("")} role="alert">{error} <small>(click to dismiss)</small></div>}

      <aside className="hud panel">
        <nav>
          {(["chat", "tasks", "comms", "knowledge"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
        {tab === "chat" && current && <Chat agent={current} messages={messages} onError={setError} langfuse={!!config?.langfuse} />}
        {tab === "tasks" && <Tasks tasks={tasks} name={name} run={run} />}
        {tab === "comms" && <Comms messages={messages} name={name} />}
        {tab === "knowledge" && <Knowledge boxes={boxes} agents={agents} run={run} />}
      </aside>

      {editing && config && settings && (
        <AgentEditor agent={editing === "new" ? null : editing} roomId={room?.id ?? ""} agents={agents} boxes={boxes} config={config}
                     tiers={settings.tiers} onClose={() => setEditing(null)} run={run} />
      )}
      {showSettings && settings && <SettingsModal settings={settings} onClose={() => setShowSettings(false)} run={run} />}
    </div>
  );
}

function Chat({ agent, messages, onError, langfuse }: { agent: Agent; messages: Message[]; onError: (e: string) => void; langfuse: boolean }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [rated, setRated] = useState<Record<string, number>>({});
  const end = useRef<HTMLDivElement>(null);
  const thread = messages.filter((m) =>
    (m.sender === "user" && m.recipient === agent.id) || (m.sender === agent.id && m.recipient === "user"));
  // Braces matter: scrollIntoView returns a Promise in newer Chromium, and an effect must not return one.
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [thread.length, agent.id]);

  const send = async () => {
    const content = text.trim();
    if (!content || busy) return;
    setText(""); setBusy(true);
    try { await api.chat(agent.id, content); } catch (e) { onError((e as Error).message); setText(content); }
    setBusy(false);
  };
  const rate = (m: Message, value: number) => {
    setRated((r) => ({ ...r, [m.id]: value }));
    api.feedback(m.id, value).catch((e) => onError(e.message));
  };

  return (
    <section className="chat">
      <div className="who"><b>{agent.name}</b><span>{agent.role}</span></div>
      <div className="thread">
        {thread.length === 0 && (
          <p className="empty">
            {agent.is_orchestrator
              ? `Ask ${agent.name} for anything. It answers, or hands the work to the right agent and reports back. Try "what is everyone working on?"`
              : `Message ${agent.name} directly for a single reply.`}
          </p>
        )}
        {thread.map((m) => (
          <div key={m.id} className={`msg ${m.sender === "user" ? "me" : ""} ${m.kind}`}>
            {m.kind === "report" && <div className="label">final report</div>}
            {m.sender === "user" ? m.content : <Markdown text={m.content} />}
            {m.sender !== "user" && langfuse && m.trace_id && (
              <div className="rate">
                <button className={rated[m.id] === 1 ? "on" : ""} onClick={() => rate(m, 1)} aria-label="Good answer">👍</button>
                <button className={rated[m.id] === 0 ? "on" : ""} onClick={() => rate(m, 0)} aria-label="Bad answer">👎</button>
              </div>
            )}
          </div>
        ))}
        {busy && <div className="msg typing">{agent.name} is working…</div>}
        <div ref={end} />
      </div>
      <div className="compose">
        <textarea value={text} placeholder={`Message ${agent.name}`} rows={2} onChange={(e) => setText(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }} />
        <button className="primary" disabled={busy || !text.trim()} onClick={send}>Send</button>
      </div>
    </section>
  );
}

function Tasks({ tasks, name, run }: { tasks: Task[]; name: (id: string) => string; run: Run }) {
  const [open, setOpen] = useState<string | null>(null);
  const sorted = [...tasks].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const roots = sorted.filter((t) => !t.parent_id);
  const row = (t: Task, depth: number): ReactElement => (
    <div key={t.id}>
      <div className="task" style={{ marginLeft: depth * 18 }} onClick={() => setOpen(open === t.id ? null : t.id)}>
        <span className={`chip s-${t.status}`}>{t.status.replace("_", " ")}</span>
        <span className="title">{t.title}</span>
        <span className="owner">{name(t.assignee_id)}{t.handoffs > 0 && ` · ${t.handoffs} handoff${t.handoffs > 1 ? "s" : ""}`}</span>
        {(t.status === "failed" || sorted.some((c) => c.parent_id === t.id && c.status === "failed")) && (
          <button className="retry" title="Run only the failed work again; finished tasks keep their results"
                  onClick={(e) => { e.stopPropagation(); run(() => api.retryTask(t.id)); }}>Retry</button>
        )}
      </div>
      {open === t.id && <div className="detail" style={{ marginLeft: depth * 18 }}><Markdown text={t.result || t.brief} /></div>}
      {sorted.filter((c) => c.parent_id === t.id).reverse().map((c) => row(c, depth + 1))}
    </div>
  );
  return (
    <section className="scroll">
      {roots.length === 0 && <p className="empty">No tasks yet. Work the orchestrator delegates shows up here, with each subtask under its mission.</p>}
      {roots.map((t) => row(t, 0))}
    </section>
  );
}

function Comms({ messages, name }: { messages: Message[]; name: (id: string) => string }) {
  const feed = messages.filter((m) => m.sender !== "user" && m.recipient !== "user").reverse();
  return (
    <section className="scroll">
      {feed.length === 0 && <p className="empty">Everything agents say to each other (delegations, handoffs, consultations, results) is logged here.</p>}
      {feed.map((m) => (
        <div key={m.id} className="comm">
          <div><b>{name(m.sender)}</b> → <b>{name(m.recipient)}</b> <span className={`chip k-${m.kind}`}>{m.kind}</span></div>
          <Markdown text={m.content} />
        </div>
      ))}
    </section>
  );
}

function Knowledge({ boxes, agents, run }: { boxes: Box[]; agents: Agent[]; run: Run }) {
  const [name, setName] = useState("");
  return (
    <section className="scroll">
      <p className="empty">A box is a private document store. An agent can only search the boxes it is assigned (set this in the agent editor).</p>
      {boxes.map((b) => (
        <div key={b.id} className="box">
          <div className="boxhead">
            <b>{b.name}</b>
            <span className="grow" />
            <label className="btn">
              Upload
              <input type="file" hidden accept=".txt,.md,.csv,.pdf,.json" onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) run(() => api.upload(b.id, f));
                e.target.value = "";
              }} />
            </label>
            <button onClick={() => confirm(`Delete "${b.name}" and its documents?`) && run(() => api.deleteBox(b.id))}>Delete</button>
          </div>
          {b.documents.map((d) => <div key={d.id} className="doc">{d.filename} <span>{d.n_chunks} chunks</span></div>)}
          {b.documents.length === 0 && <div className="doc"><span>empty</span></div>}
          <div className="access">
            Access: {agents.filter((a) => a.box_ids.includes(b.id)).map((a) => <span key={a.id} className="chip">{a.name}</span>)}
            {!agents.some((a) => a.box_ids.includes(b.id)) && <span>nobody yet</span>}
          </div>
        </div>
      ))}
      <div className="compose inline">
        <input value={name} placeholder="New box name" onChange={(e) => setName(e.target.value)} />
        <button disabled={!name.trim()} onClick={() => { run(() => api.createBox(name.trim(), "")); setName(""); }}>Create</button>
      </div>
    </section>
  );
}

function AgentEditor({ agent, roomId, agents, boxes, config, tiers, onClose, run }: {
  agent: Agent | null; roomId: string; agents: Agent[]; boxes: Box[]; config: Config; tiers: Record<string, string>; onClose: () => void; run: Run;
}) {
  const [f, setF] = useState({
    name: agent?.name ?? "", role: agent?.role ?? "", instructions: agent?.instructions ?? "", tier: agent?.tier ?? "auto",
    model: agent?.model ?? "", parent_id: agent?.parent_id ?? "", tools: agent?.tools ?? [], box_ids: agent?.box_ids ?? [],
  });
  const set = <K extends keyof typeof f>(k: K, v: (typeof f)[K]) => setF((cur) => ({ ...cur, [k]: v }));
  const toggle = (k: "tools" | "box_ids", v: string) => set(k, f[k].includes(v) ? f[k].filter((x) => x !== v) : [...f[k], v]);
  const orch = agent?.is_orchestrator;
  const save = () => { run(() => api.saveAgent(agent?.id ?? null, agent ? f : { ...f, room_id: roomId })); onClose(); };

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <h2>{agent ? `Edit ${agent.name}` : "New agent"}</h2>
        <label>Name<input value={f.name} onChange={(e) => set("name", e.target.value)} autoFocus /></label>
        <label>Role <small>one line; the orchestrator routes work by this</small>
          <input value={f.role} onChange={(e) => set("role", e.target.value)} placeholder="Sales rep for the East region" /></label>
        <label>Instructions <small>how it should behave</small>
          <textarea rows={4} value={f.instructions} onChange={(e) => set("instructions", e.target.value)} /></label>
        <div className="cols">
          <label>Model tier
            <select value={f.tier} onChange={(e) => set("tier", e.target.value)}>
              <option value="auto">auto (by task difficulty)</option>
              {TIERS.map((t) => <option key={t} value={t}>{t} · {tiers[t]}</option>)}
            </select></label>
          <label>Model override <small>any OpenRouter id</small>
            <input value={f.model} onChange={(e) => set("model", e.target.value)} placeholder="openai/gpt-5.6-terra" /></label>
        </div>
        {!orch && (
          <label>Reports to
            <select value={f.parent_id} onChange={(e) => set("parent_id", e.target.value)}>
              <option value="">the orchestrator</option>
              {agents.filter((a) => !a.is_orchestrator && a.id !== agent?.id).map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
            </select></label>
        )}
        <fieldset><legend>Tools</legend>
          {config.optional_tools.map((t) => (
            <label key={t} className="check"><input type="checkbox" checked={f.tools.includes(t)} onChange={() => toggle("tools", t)} />{t}</label>
          ))}
        </fieldset>
        <fieldset><legend>Knowledge boxes</legend>
          {boxes.length === 0 && <small>Create a box in the Knowledge tab first.</small>}
          {boxes.map((b) => (
            <label key={b.id} className="check"><input type="checkbox" checked={f.box_ids.includes(b.id)} onChange={() => toggle("box_ids", b.id)} />{b.name}</label>
          ))}
        </fieldset>
        <div className="actions">
          {agent && !orch && <button className="danger" onClick={() => { if (confirm(`Delete ${agent.name}?`)) { run(() => api.deleteAgent(agent.id)); onClose(); } }}>Delete</button>}
          <span className="grow" />
          <button onClick={onClose}>Cancel</button>
          <button className="primary" disabled={!f.name.trim()} onClick={save}>Save</button>
        </div>
      </div>
    </div>
  );
}

const usd = (n: number | null) => (n == null ? "–" : n < 0.01 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`);

function StatTable({ title, rows }: { title: string; rows: StatRow[] }) {
  return (
    <table className="stats">
      <thead><tr><th>{title}</th><th>runs</th><th>judged</th><th>avg score</th><th>avg cost</th><th>total</th></tr></thead>
      <tbody>
        {rows.length === 0 && <tr><td colSpan={6}>No runs yet.</td></tr>}
        {rows.map((r) => (
          <tr key={r.key}><td>{r.key}</td><td>{r.runs}</td><td>{r.judged}</td>
            <td>{r.avg_score == null ? "–" : r.avg_score.toFixed(2)}</td><td>{usd(r.avg_cost)}</td><td>{usd(r.total_cost)}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

function SettingsModal({ settings, onClose, run }: { settings: Settings; onClose: () => void; run: Run }) {
  const [key, setKey] = useState("");
  const [tiers, setTiers] = useState(settings.tiers);
  const [models, setModels] = useState<Model[]>([]);
  const [stats, setStats] = useState<RouterStats | null>(null);
  const [note, setNote] = useState("");

  const loadStats = useCallback(() => { api.routerStats().then(setStats).catch(() => {}); }, []);
  useEffect(() => { api.models().then(setModels).catch(() => {}); loadStats(); }, [loadStats]);

  const saveKey = async () => { await run(() => api.saveSettings({ openrouter_key: key.trim() })); setKey(""); };
  const saveTiers = () => run(() => api.saveSettings({ tier_fast: tiers.fast, tier_balanced: tiers.balanced, tier_smart: tiers.smart }));
  const train = async () => {
    try {
      const r = await api.trainRouter();
      setNote(r.usable
        ? `Trained on ${r.samples} judged runs. Switch routing to "learned" to use it.`
        : `Trained on ${r.samples} judged runs, but it needs at least ${stats?.min_samples ?? 15} on both the fast and balanced tiers before it can route.`);
      loadStats();
    } catch (e) { setNote((e as Error).message); }
  };

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal wide" onClick={(e) => e.stopPropagation()}>
        <h2>Settings</h2>

        <h3>OpenRouter API key</h3>
        <p className="help">
          Your agents run on your own key. It is encrypted on the server and never shown again.
          Get one at <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">openrouter.ai/keys</a>.
        </p>
        <div className="row">
          <input type="password" value={key} autoComplete="off" onChange={(e) => setKey(e.target.value)}
                 placeholder={settings.has_key ? `saved · ends in ${settings.key_hint}` : "sk-or-…"} aria-label="OpenRouter API key" />
          <button className="primary" disabled={!key.trim()} onClick={saveKey}>Save key</button>
          {settings.has_key && <button className="danger" onClick={() => run(() => api.saveSettings({ openrouter_key: "" }))}>Remove</button>}
        </div>

        <h3>Models per tier</h3>
        <p className="help">Each agent uses one of these tiers, or "auto" to let the router pick per task. Any tool-capable OpenRouter model works.</p>
        <datalist id="models">
          {models.map((m) => <option key={m.id} value={m.id}>{`${m.name} · $${m.input.toFixed(2)} in / $${m.output.toFixed(2)} out per 1M`}</option>)}
        </datalist>
        <div className="cols three">
          {TIERS.map((t) => (
            <label key={t}>{t}<input list="models" value={tiers[t]} onChange={(e) => setTiers({ ...tiers, [t]: e.target.value })} /></label>
          ))}
        </div>
        <div className="row"><span className="grow" /><button onClick={saveTiers}>Save models</button></div>

        <h3>Quality scoring and routing</h3>
        <label className="check block">
          <input type="checkbox" checked={settings.judge_enabled} onChange={(e) => run(() => api.saveSettings({ judge_enabled: e.target.checked }))} />
          Score every finished task with an LLM judge <small>(one extra balanced-tier call per task; this is what the learned router trains on)</small>
        </label>
        <div className="cols">
          <label>Routing for "auto" agents
            <select value={settings.router_mode} onChange={(e) => run(() => api.saveSettings({ router_mode: e.target.value as Settings["router_mode"] }))}>
              <option value="heuristic">heuristic (keyword and length rules)</option>
              <option value="learned" disabled={!stats?.usable}>learned{stats?.usable ? "" : " (train it first)"}</option>
            </select></label>
          <label>Exploration <small>{Math.round(settings.explore_rate * 100)}% of auto tasks try a random tier</small>
            <input type="range" min={0} max={0.5} step={0.05} defaultValue={settings.explore_rate}
                   onMouseUp={(e) => run(() => api.saveSettings({ explore_rate: Number((e.target as HTMLInputElement).value) }))}
                   onKeyUp={(e) => run(() => api.saveSettings({ explore_rate: Number((e.target as HTMLInputElement).value) }))} /></label>
        </div>
        {stats && <StatTable title="tier" rows={stats.by_tier} />}
        {stats && <StatTable title="routed by" rows={stats.by_router} />}
        <div className="row">
          <span className="help grow">{note || `The learned router needs ${stats?.min_samples ?? 15}+ judged task runs on the fast and balanced tiers.`}</span>
          <button onClick={train}>Train router</button>
        </div>

        <div className="actions"><span className="grow" /><button className="primary" onClick={onClose}>Done</button></div>
      </div>
    </div>
  );
}
