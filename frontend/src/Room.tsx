import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { Agent, AgentState } from "./api";

export interface Packet { id: string; from: string; to: string; kind: string }
type Point = { x: number; y: number };

/** Where everyone's desk is, in percent of the room. The orchestrator sits at the back, each
 *  reporting level one row closer to the front, every lead centred over its reports, and "you"
 *  stand at the front door. */
export function layout(agents: Agent[]): Record<string, Point> {
  const orch = agents.find((a) => a.is_orchestrator);
  if (!orch) return {};
  const ids = new Set(agents.map((a) => a.id));
  const parentOf = (a: Agent) => (a.parent_id && ids.has(a.parent_id) ? a.parent_id : orch.id);
  const kids = (id: string) => agents.filter((a) => !a.is_orchestrator && parentOf(a) === id);

  const raw: Record<string, { col: number; depth: number }> = {};
  let next = 0;
  let maxDepth = 0;
  const place = (a: Agent, depth: number): number => {
    maxDepth = Math.max(maxDepth, depth);
    const cols = kids(a.id).map((c) => place(c, depth + 1));
    const col = cols.length ? (cols[0] + cols[cols.length - 1]) / 2 : next++;
    raw[a.id] = { col, depth };
    return col;
  };
  place(orch, 0);

  const cols = Math.max(next, 1);
  const rowGap = Math.min(17, 36 / Math.max(maxDepth, 1));
  const out: Record<string, Point> = { user: { x: 50, y: 95 } };
  for (const [id, p] of Object.entries(raw)) out[id] = { x: 6 + ((p.col + 0.5) / cols) * 88, y: 45 + p.depth * rowGap };
  return out;
}

function hue(name: string): number {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360;
  return h;
}

/** The character. An original little robot: body colour from the agent's name, face on a visor. */
export function Buddy({ hue: h, state, boss = false }: { hue: number; state: AgentState; boss?: boolean }) {
  const ink = "#1d1b3a";
  const body = `hsl(${h} 85% 64%)`;
  const dark = `hsl(${h} 70% 46%)`;
  const asleep = state === "idle";
  return (
    <svg className={`buddy ${state}`} viewBox="0 0 80 100" aria-hidden="true">
      <ellipse cx="40" cy="94" rx="24" ry="5" fill="#1d1b3a" opacity=".22" />
      <g className="buddy-body">
        <ellipse cx="29" cy="89" rx="9" ry="5.5" fill={dark} stroke={ink} strokeWidth="2.5" />
        <ellipse cx="51" cy="89" rx="9" ry="5.5" fill={dark} stroke={ink} strokeWidth="2.5" />
        <rect className="arm arm-l" x="6" y="52" width="12" height="22" rx="6" fill={dark} stroke={ink} strokeWidth="2.5" />
        <rect className="arm arm-r" x="62" y="52" width="12" height="22" rx="6" fill={dark} stroke={ink} strokeWidth="2.5" />
        <rect x="13" y="30" width="54" height="58" rx="25" fill={body} stroke={ink} strokeWidth="3" />
        <ellipse cx="40" cy="70" rx="15" ry="12" fill="#fff" opacity=".4" />
        <rect x="20" y="37" width="40" height="24" rx="12" fill={ink} />
        {asleep ? (
          <g stroke="#9fe8ff" strokeWidth="2.5" strokeLinecap="round" fill="none">
            <path d="M27 49 q4 3.5 8 0" /><path d="M45 49 q4 3.5 8 0" />
          </g>
        ) : (
          <g className="eyes" fill="#9fe8ff">
            <circle cx="31" cy="48" r="4.5" /><circle cx="49" cy="48" r="4.5" />
          </g>
        )}
        {state === "talking" && <ellipse className="mouth" cx="40" cy="56" rx="3.5" ry="2" fill="#ff8fb1" />}
        {boss ? (
          <path d="M26 31 l-3 -15 9 7 8 -11 8 11 9 -7 -3 15 z" fill="#ffd23f" stroke={ink} strokeWidth="2.5" strokeLinejoin="round" />
        ) : (
          <g>
            <line x1="40" y1="30" x2="40" y2="18" stroke={ink} strokeWidth="3" />
            <circle className="bulb" cx="40" cy="14" r="5.5" fill={asleep ? "#8d8fb5" : "#ffd23f"} stroke={ink} strokeWidth="2.5" />
          </g>
        )}
      </g>
    </svg>
  );
}

function Desk() {
  return (
    <svg className="desk" viewBox="0 0 110 50" aria-hidden="true">
      <rect x="4" y="18" width="102" height="14" rx="7" fill="#ffc98b" stroke="#1d1b3a" strokeWidth="3" />
      <rect x="14" y="30" width="8" height="16" rx="3" fill="#e0a25e" stroke="#1d1b3a" strokeWidth="2.5" />
      <rect x="88" y="30" width="8" height="16" rx="3" fill="#e0a25e" stroke="#1d1b3a" strokeWidth="2.5" />
      <rect x="40" y="2" width="30" height="19" rx="4" fill="#2b2d6e" stroke="#1d1b3a" strokeWidth="2.5" />
      <rect x="44" y="6" width="22" height="11" rx="2" fill="#7df0d8" opacity=".85" />
    </svg>
  );
}

function statusText(a: Agent): string {
  if (a.state === "idle") return "napping";
  if (a.state === "tool") return `using ${a.detail}`;
  if (a.state === "talking") return a.detail || "chatting";
  return "thinking";
}

/** One agent on the floor. It lives at its desk and walks over to whoever it is talking to. */
function Walker({ agent, home, target, bubble, selected, onSelect, onEdit }: {
  agent: Agent; home: Point; target: Point; bubble?: string; selected: boolean; onSelect: () => void; onEdit: () => void;
}) {
  const [walking, setWalking] = useState(false);
  const [flip, setFlip] = useState(false);
  const last = useRef(target);
  useEffect(() => {
    if (last.current.x === target.x && last.current.y === target.y) return;
    setFlip(target.x < last.current.x);
    last.current = target;
    setWalking(true);
    const t = setTimeout(() => setWalking(false), 850);
    return () => clearTimeout(t);
  }, [target.x, target.y]);

  // Layering: the walker itself is a zero-size anchor at the agent's feet with no stacking context,
  // so its parts sort against the desks individually: sprite behind its desk, name tag in front.
  const z = Math.round(target.y * 10);
  void home;
  return (
    <div className={`walker ${agent.state} ${walking ? "walking" : ""} ${selected ? "sel" : ""} ${agent.is_orchestrator ? "boss" : ""}`}
         style={{ left: `${target.x}%`, top: `${target.y}%` }}
         onClick={onSelect} title={agent.role} role="button" tabIndex={0}
         onKeyDown={(e) => e.key === "Enter" && onSelect()}>
      {bubble && <div className="speech">{bubble}</div>}
      {!bubble && agent.state === "thinking" && <div className="thought"><i /><i /><i /></div>}
      {!bubble && agent.state === "tool" && <div className="toolchip">🛠 {agent.detail}</div>}
      {agent.state === "idle" && <div className="zzz"><i>z</i><i>z</i><i>Z</i></div>}
      <div className={`sprite ${flip ? "flip" : ""}`} style={{ zIndex: z }}>
        <Buddy hue={agent.is_orchestrator ? 42 : hue(agent.name)} state={agent.state} boss={agent.is_orchestrator} />
      </div>
      <div className="tag" style={{ zIndex: z + 2 }}>
        <b>{agent.name}</b>
        <button className="edit" onClick={(e) => { e.stopPropagation(); onEdit(); }} aria-label={`Edit ${agent.name}`}>✎</button>
        <span>{statusText(agent)}</span>
      </div>
    </div>
  );
}

interface Props {
  agents: Agent[];
  bubbles: Record<string, string>;
  packets: Packet[];
  selected: string | null;
  openTasks: number;
  onSelect: (id: string) => void;
  onEdit: (a: Agent) => void;
}

export default function Room({ agents, bubbles, packets, selected, openTasks, onSelect, onEdit }: Props) {
  const home = layout(agents);
  const orch = agents.find((a) => a.is_orchestrator);
  const awake = agents.filter((a) => a.state !== "idle").length;

  // While talking, an agent stands next to its peer's desk instead of its own.
  const spot = (a: Agent): Point => {
    const peer = a.state === "talking" && a.peer ? home[a.peer] : null;
    if (!peer) return home[a.id];
    const side = home[a.id].x <= peer.x ? -1 : 1;
    return { x: peer.x + side * 7, y: peer.y + 1.5 };
  };

  return (
    <div className="room">
      <div className="wall">
        <div className="window"><i className="sun" /><i className="hill h1" /><i className="hill h2" /></div>
        <div className="board">
          <small>mission board</small>
          <b>{openTasks}</b><span>open {openTasks === 1 ? "task" : "tasks"}</span>
          <b>{awake}</b><span>awake</span>
        </div>
        <div className="window"><i className="sun moon" /><i className="hill h1" /><i className="hill h2" /></div>
      </div>
      <div className="floor"><div className="tiles" /></div>
      <div className="rug" />

      <svg className="paths" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        {orch && agents.filter((a) => !a.is_orchestrator).map((a) => {
          const p = home[a.parent_id && home[a.parent_id] ? a.parent_id : orch.id];
          return <line key={a.id} x1={p.x} y1={p.y} x2={home[a.id].x} y2={home[a.id].y} />;
        })}
      </svg>

      {agents.map((a) => home[a.id] && (
        <div key={`desk-${a.id}`} className="deskspot" style={{ left: `${home[a.id].x}%`, top: `${home[a.id].y}%`, zIndex: Math.round(home[a.id].y * 10) + 1 }}>
          <Desk />
        </div>
      ))}

      {agents.map((a) => home[a.id] && (
        <Walker key={a.id} agent={a} home={home[a.id]} target={spot(a)} bubble={bubbles[a.id]}
                selected={selected === a.id} onSelect={() => onSelect(a.id)} onEdit={() => onEdit(a)} />
      ))}

      <div className="you" style={{ left: `${home.user?.x ?? 50}%`, top: `${home.user?.y ?? 95}%` }}>You</div>

      {packets.map((p) => home[p.from] && home[p.to] && (
        <div key={p.id} className={`letter k-${p.kind}`} style={{
          "--x1": `${home[p.from].x}%`, "--y1": `${home[p.from].y - 6}%`,
          "--x2": `${home[p.to].x}%`, "--y2": `${home[p.to].y - 6}%`,
        } as CSSProperties} />
      ))}

      {agents.length === 1 && (
        <div className="hint">It's quiet in here. Add an agent, or load the demo team, to fill the room.</div>
      )}
    </div>
  );
}
