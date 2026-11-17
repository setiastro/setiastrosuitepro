# src/setiastro/saspro/workflow_graph.py
# Copyright (c) Franklin Marek / SetiAstro (www.setiastro.com). GPLv3.
"""
Graph view of a Workflow Assistant workflow.

Lanes are columns and steps flow downward. A step's row is its depth: the
longest chain of lane order and links that leads to it. So a branch starts
below the step it branches from, and a join sits below every step that feeds
it. Links are the only edges that cross lanes.

The view never owns workflow state. It reads the host dialog's workflow,
selection and completion through _WorkflowGraphHost (workflows.py), and every
edit goes back through the host, which reuses the dialog's own commands. The
structure and placement functions at the top are pure so they can be tested
without a display.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QEvent, QPointF, QRect, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
    QPolygonF,
    QTransform,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from setiastro.saspro.workflows import (
    LINK_DROP_REASONS,
    WORKFLOW_MIME,
    WorkflowDefinition,
    WorkflowLink,
    WorkflowStep,
    WorkflowToolTree,
    _summarize_preset,
    format_workflow_step_label,
    normalize_workflow_links,
)

# -----------------------------------------------------------------------------
# Geometry (scene units = pixels at 100% zoom)
# -----------------------------------------------------------------------------

MARGIN = 18
HEADER_H = 30
HEADER_GAP = 14
ROW_GAP = 24          # between rows joined only by lane order
LINK_GAP = 58         # between rows a link crosses, so its curve and label have room
COL_GAP = 44
LANE_PAD = 14
NODE_MIN_W = 160
NODE_MAX_W = 240
ACTION_H = 32
NOTE_H = 46
MARKER_H = 26
CHOICE_HEAD_H = 26
CHOICE_ROW_H = 22
SLOT_H = 42
PORT_R = 5.0
NEW_LANE_W = 150

ZOOM_MIN = 0.25
ZOOM_MAX = 2.5
FIT_MIN = 0.7    # Fit shrinks only while text stays readable; past that it scrolls at 100%.

SLOT_PREFIX = "lane:"


def slot_key(lane: str) -> str:
    """Node key of the drop slot shown in an empty lane."""
    return SLOT_PREFIX + lane


# -----------------------------------------------------------------------------
# Structure and placement (pure)
# -----------------------------------------------------------------------------

@dataclass
class GraphNode:
    key: str                      # step id, or slot_key(lane) for an empty lane
    lane: str
    kind: str                     # action, note, split, merge, choice, slot
    step: Optional[WorkflowStep] = None
    children: List[WorkflowStep] = field(default_factory=list)   # a choice's actions
    rank: int = 0
    pinned: bool = False          # the user dragged it to its row
    x: float = 0.0
    y: float = 0.0
    w: float = float(NODE_MIN_W)
    h: float = float(ACTION_H)

    @property
    def is_slot(self) -> bool:
        return self.kind == "slot"


@dataclass
class GraphEdge:
    src: str
    dst: str
    link: Optional[WorkflowLink] = None   # None: the lane's own order
    label: str = ""                       # I/O tags shared by both ends


@dataclass
class WorkflowGraph:
    lanes: List[str]
    nodes: Dict[str, GraphNode]
    edges: List[GraphEdge]
    order: Dict[str, List[str]]           # node keys per lane, top to bottom
    columns: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # lane -> (x, width)
    row_top: List[float] = field(default_factory=list)     # per row: top y, height, gap below
    row_h: List[float] = field(default_factory=list)
    row_gap: List[float] = field(default_factory=list)
    new_lane_x: float = 0.0
    width: float = 0.0
    height: float = 0.0


def _link_label(nodes: Dict[str, GraphNode], order: Dict[str, List[str]],
                src: str, dst: str, to_lane: bool) -> str:
    step = nodes[src].step
    if step is None or not step.outputs:
        return ""
    if to_lane:
        lane_steps = [nodes[k].step for k in order.get(nodes[dst].lane, []) if nodes[k].step]
        inputs = next((s.inputs for s in lane_steps if s.inputs), [])
    else:
        inputs = nodes[dst].step.inputs if nodes[dst].step else []
    return ", ".join(t for t in step.outputs if t in inputs)


def _assign_ranks(nodes: Dict[str, GraphNode], edges: List[GraphEdge],
                  lanes: List[str], order: Dict[str, List[str]],
                  pins: Optional[Dict[str, int]] = None) -> None:
    """Longest-path depth from the lane heads (Kahn order).

    A pinned step takes its pinned row and ignores its incoming links, but it
    is never placed above the step before it in its own lane. Everything
    after it is ranked from where it ends up.
    """
    pins = pins or {}
    succ: Dict[str, List[Tuple[str, bool]]] = {k: [] for k in nodes}
    indeg = {k: 0 for k in nodes}
    for e in edges:
        succ[e.src].append((e.dst, e.link is not None))
        indeg[e.dst] += 1
    for n in nodes.values():
        n.rank = 0

    def settle(k: str):            # every input of k is in: apply its pin
        if k in pins:
            nodes[k].rank = max(nodes[k].rank, pins[k])

    queue = [k for lane in lanes for k in order[lane] if indeg[k] == 0]
    for k in queue:
        settle(k)
    done = set()
    i = 0
    while i < len(queue):
        k = queue[i]
        i += 1
        done.add(k)
        for d, is_link in succ[k]:
            if not (is_link and d in pins):
                nodes[d].rank = max(nodes[d].rank, nodes[k].rank + 1)
            indeg[d] -= 1
            if indeg[d] == 0:
                settle(d)
                queue.append(d)
    if len(done) < len(nodes):
        # Links are normalized loop-free, so this is only a guard: stack
        # whatever is left below everything else so nothing overlaps.
        nxt = max((nodes[k].rank for k in done), default=-1) + 1
        for lane in lanes:
            for k in order[lane]:
                if k not in done:
                    nodes[k].rank = nxt
                    nxt += 1


def build_workflow_graph(wf: WorkflowDefinition, rows: Optional[Dict[str, int]] = None) -> WorkflowGraph:
    """Nodes, edges and ranks for a workflow, before any geometry.

    rows overrides wf.graph_rows (the user's pins), e.g. to ask where a step
    would sit on its own.
    """
    if rows is None:
        rows = getattr(wf, "graph_rows", None) or {}
    lanes = list(wf.lanes) or ["main"]
    known = set(lanes)
    kids: Dict[str, List[WorkflowStep]] = {}
    for s in wf.steps:
        if s.parent_id:
            kids.setdefault(s.parent_id, []).append(s)

    nodes: Dict[str, GraphNode] = {}
    order: Dict[str, List[str]] = {lane: [] for lane in lanes}
    for s in wf.steps:
        if s.parent_id:
            continue
        lane = s.lane if s.lane in known else lanes[0]
        nodes[s.id] = GraphNode(
            key=s.id, lane=lane, kind=s.kind or "action", step=s,
            children=list(kids.get(s.id, [])) if s.kind == "choice" else [],
        )
        order[lane].append(s.id)
    for lane in lanes:
        if not order[lane]:
            key = slot_key(lane)
            nodes[key] = GraphNode(key=key, lane=lane, kind="slot", h=float(SLOT_H))
            order[lane].append(key)

    edges: List[GraphEdge] = []
    for lane in lanes:
        seq = order[lane]
        edges.extend(GraphEdge(src=a, dst=b) for a, b in zip(seq, seq[1:]))
    for link in wf.links or []:
        if link.src not in nodes:
            continue
        if link.dst:
            dst = link.dst if link.dst in nodes else ""
        else:
            seq = order.get(link.dst_lane) or []
            dst = seq[0] if seq else ""
        if not dst or dst == link.src or nodes[dst].lane == nodes[link.src].lane:
            continue
        edges.append(GraphEdge(
            src=link.src, dst=dst, link=link,
            label=_link_label(nodes, order, link.src, dst, to_lane=not link.dst),
        ))

    pins: Dict[str, int] = {}
    for key, row in rows.items():
        n = nodes.get(key)
        if n is not None and not n.is_slot and isinstance(row, int) and not isinstance(row, bool) and row >= 0:
            pins[key] = row
            n.pinned = True
    _assign_ranks(nodes, edges, lanes, order, pins)
    return WorkflowGraph(lanes=lanes, nodes=nodes, edges=edges, order=order)


def natural_rank(wf: WorkflowDefinition, key: str) -> int:
    """The row a step would take if it were not pinned (other pins still count)."""
    rows = dict(getattr(wf, "graph_rows", None) or {})
    rows.pop(key, None)
    n = build_workflow_graph(wf, rows).nodes.get(key)
    return n.rank if n is not None else 0


def row_at_y(g: WorkflowGraph, y: float) -> int:
    """Row whose band holds y. Each band runs halfway into the gaps on either side;
    below the last row is one new row."""
    for r in range(len(g.row_top)):
        if y < g.row_top[r] + g.row_h[r] + g.row_gap[r] / 2:
            return r
    return len(g.row_top)


def row_top_y(g: WorkflowGraph, row: int) -> float:
    if 0 <= row < len(g.row_top):
        return g.row_top[row]
    if not g.row_top:
        return float(MARGIN + HEADER_H + HEADER_GAP)
    return g.row_top[-1] + g.row_h[-1] + ROW_GAP + (row - len(g.row_top)) * (ACTION_H + ROW_GAP)


def row_height(g: WorkflowGraph, row: int) -> float:
    return g.row_h[row] if 0 <= row < len(g.row_h) else float(ACTION_H)


PILL_KINDS = ("split", "merge", "slot")


def place_workflow_graph(g: WorkflowGraph, sizes: Dict[str, Tuple[float, float]],
                         reserve_new_lane: bool = False) -> WorkflowGraph:
    """Give every node a position and final size.

    A lane column fits its widest node, and every box in it takes that width
    (split/merge pills and empty-lane slots keep their own). A row fits its
    tallest node. The gap under a row widens when a link drops into the very
    next row, so the curve is not squashed flat.
    """
    for key, n in g.nodes.items():
        if key in sizes:
            n.w, n.h = float(sizes[key][0]), float(sizes[key][1])

    x = float(MARGIN)
    for lane in g.lanes:
        inner = max([g.nodes[k].w for k in g.order[lane]] + [float(NODE_MIN_W)])
        for k in g.order[lane]:
            if g.nodes[k].kind not in PILL_KINDS:
                g.nodes[k].w = inner
        width = inner + 2 * LANE_PAD
        g.columns[lane] = (x, width)
        x += width + COL_GAP
    g.new_lane_x = x
    if reserve_new_lane:
        x += NEW_LANE_W + COL_GAP
    g.width = x - COL_GAP + MARGIN

    top_rank = max((n.rank for n in g.nodes.values()), default=0)
    row_h = [0.0] * (top_rank + 1)
    for n in g.nodes.values():
        row_h[n.rank] = max(row_h[n.rank], n.h)
    # A row left empty by a pin keeps a step's height, so the grid stays even.
    row_h = [h if h > 0 else float(ACTION_H) for h in row_h]
    gap = [float(ROW_GAP)] * (top_rank + 1)
    for e in g.edges:
        if e.link is not None and g.nodes[e.dst].rank == g.nodes[e.src].rank + 1:
            gap[g.nodes[e.src].rank] = float(LINK_GAP)
    row_y = []
    y = float(MARGIN + HEADER_H + HEADER_GAP)
    for r, h in enumerate(row_h):
        row_y.append(y)
        y += h + gap[r]
    g.height = y - gap[-1] + LANE_PAD + MARGIN
    g.row_top, g.row_h, g.row_gap = row_y, row_h, gap

    for n in g.nodes.values():
        cx, cw = g.columns[n.lane]
        n.x = cx + (cw - n.w) / 2
        n.y = row_y[n.rank] + (row_h[n.rank] - n.h) / 2
    return g


# -----------------------------------------------------------------------------
# Drop payloads (same formats the lane lists accept)
# -----------------------------------------------------------------------------

def _shortcut_mime() -> Optional[str]:
    try:
        from setiastro.saspro.dnd_mime import MIME_CMD
        return MIME_CMD
    except Exception:
        try:
            from setiastro.saspro.shortcuts import MIME_CMD
            return MIME_CMD
        except Exception:
            return None


def has_tool_payload(md) -> bool:
    mc = _shortcut_mime()
    return bool(md is not None and (md.hasFormat(WORKFLOW_MIME) or (mc and md.hasFormat(mc))))


def tool_payloads(md) -> List[dict]:
    """[{command_id, preset, name}] for a catalog drag, a shortcut, or an expanded bundle."""
    if md is None:
        return []
    if md.hasFormat(WORKFLOW_MIME):
        try:
            cid = bytes(md.data(WORKFLOW_MIME)).decode("utf-8").strip()
        except Exception:
            cid = ""
        return [{"command_id": cid, "preset": {}, "name": ""}] if cid else []

    mc = _shortcut_mime()
    if not (mc and md.hasFormat(mc)):
        return []
    try:
        from setiastro.saspro.shortcuts import _unpack_cmd_payload
        payload = _unpack_cmd_payload(bytes(md.data(mc)))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    cid = str(payload.get("command_id") or payload.get("cid") or "")
    if cid == "function_bundle":
        out = []
        for st in payload.get("steps") or []:
            if not isinstance(st, dict):
                continue
            scid = str(st.get("command_id") or st.get("cid") or "")
            if not scid or scid == "function_bundle":
                continue
            sp = st.get("preset")
            out.append({"command_id": scid, "preset": sp if isinstance(sp, dict) else {},
                        "name": str(st.get("name") or "")})
        return out
    if not cid:
        return []
    p = payload.get("preset")
    return [{"command_id": cid, "preset": p if isinstance(p, dict) else {},
             "name": str(payload.get("name") or "")}]


# -----------------------------------------------------------------------------
# Painting helpers
# -----------------------------------------------------------------------------

_LANE_HUES = (199, 28, 140, 275, 350, 50, 175, 310)


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class _Theme:
    """Colors derived from the widget palette, so every SASpro theme works."""

    def __init__(self, widget: QWidget):
        pal = widget.palette()
        self.base = pal.color(QPalette.ColorRole.Base)
        self.text = pal.color(QPalette.ColorRole.Text)
        self.hl = pal.color(QPalette.ColorRole.Highlight)
        self.dark = self.base.lightness() < 128
        self.dim = _mix(self.text, self.base, 0.45)
        self.node = _mix(self.base, self.text, 0.09 if self.dark else 0.0)
        self.node_hover = _mix(self.base, self.text, 0.16 if self.dark else 0.06)
        self.border = _mix(self.base, self.text, 0.32)
        self.done = QColor(80, 180, 95)
        self.note = QColor(212, 168, 40)
        self.choice = QColor(61, 191, 159)

    def lane(self, index: int) -> QColor:
        h = _LANE_HUES[index % len(_LANE_HUES)]
        return QColor.fromHsv(h, 140, 230) if self.dark else QColor.fromHsv(h, 200, 180)


def _wrap_lines(fm: QFontMetricsF, text: str, width: float, max_lines: int = 2) -> List[str]:
    words = text.split()
    lines: List[str] = []
    cur = ""
    for i, word in enumerate(words):
        trial = f"{cur} {word}".strip()
        if not cur or fm.horizontalAdvance(trial) <= width:
            cur = trial
            continue
        lines.append(fm.elidedText(cur, Qt.TextElideMode.ElideRight, width))
        cur = word
        if len(lines) == max_lines - 1:
            rest = " ".join(words[i:])
            lines.append(fm.elidedText(rest, Qt.TextElideMode.ElideRight, width))
            return lines
    if cur:
        lines.append(fm.elidedText(cur, Qt.TextElideMode.ElideRight, width))
    return lines


def _icon_for(step: Optional[WorkflowStep], catalog) -> QIcon:
    if step is None or step.kind != "action":
        return QIcon()
    info = catalog.get(step.command_id) if catalog else None
    icon = getattr(info, "icon", None)
    return icon if isinstance(icon, QIcon) and not icon.isNull() else QIcon()


def _node_tooltip(node: GraphNode, catalog, editable: bool) -> str:
    s = node.step
    if s is None:
        return f"Lane <b>{html.escape(node.lane)}</b> is empty." + (" Drag tools here." if editable else "")
    parts = [f"<b>{html.escape(format_workflow_step_label(s, catalog))}</b>"]
    if s.kind == "action":
        info = catalog.get(s.command_id) if catalog else None
        tip = getattr(info, "status_tip", "") or ""
        if tip:
            parts.append(html.escape(tip))
    if s.note and s.kind != "note":
        parts.append(html.escape(s.note))
    elif s.kind == "note":
        parts.append(html.escape(s.note or ""))
    if s.inputs:
        parts.append("<i>Inputs:</i> " + html.escape(", ".join(s.inputs)))
    if s.outputs:
        parts.append("<i>Outputs:</i> " + html.escape(", ".join(s.outputs)))
    if s.preset:
        parts.append("<i>Preset:</i> " + _summarize_preset(s.preset))
    if node.pinned:
        parts.append("<i>Placed by hand.</i>" + (" Right-click › Auto Position to undo." if editable else ""))
    return "<br>".join(p for p in parts if p)


# -----------------------------------------------------------------------------
# Scene items
# -----------------------------------------------------------------------------

class _NodeItem(QGraphicsItem):
    """One top-level step (a choice draws its actions as rows), or an empty lane's slot."""

    def __init__(self, node: GraphNode, view: "WorkflowGraphView", color: QColor, catalog):
        super().__init__()
        self.node = node
        self.view = view
        self.color = color
        self.done = False
        self.current = False
        self.current_row = ""
        self.hover = False
        self.port_hover = False
        self.target = False
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        fm = QFontMetricsF(view.font())
        self.icon = QIcon()
        self.rows: List[Tuple[str, str, QIcon]] = []
        kind = node.kind
        step = node.step
        if kind == "slot":
            self.label = "Drop tools here" if view.editable else "Empty lane"
            w, h = NODE_MIN_W, SLOT_H
        elif kind == "choice":
            self.label = format_workflow_step_label(step, catalog)
            for child in node.children:
                self.rows.append((child.id, format_workflow_step_label(child, catalog),
                                  _icon_for(child, catalog)))
            widest = max([fm.horizontalAdvance(self.label) + 40]
                         + [fm.horizontalAdvance(t) + 44 for _cid, t, _ic in self.rows])
            w = _clamp(widest, NODE_MIN_W, NODE_MAX_W)
            h = CHOICE_HEAD_H + max(1, len(self.rows)) * CHOICE_ROW_H + 6
        elif kind == "note":
            self.label = (step.note or "").strip() or "Note"
            w = _clamp(fm.horizontalAdvance(self.label) / 2 + 52, NODE_MIN_W, NODE_MAX_W)
            h = NOTE_H
        elif kind in ("split", "merge"):
            self.label = format_workflow_step_label(step, catalog)
            w = _clamp(fm.horizontalAdvance(self.label) + 44, 110, NODE_MAX_W)
            h = MARKER_H
        else:
            self.label = format_workflow_step_label(step, catalog)
            self.icon = _icon_for(step, catalog)
            w = _clamp(fm.horizontalAdvance(self.label) + 66, NODE_MIN_W, NODE_MAX_W)
            h = ACTION_H
        self.w, self.h = float(w), float(h)
        self.setToolTip(_node_tooltip(node, catalog, view.editable))

    # ---- geometry -----------------------------------------------------

    def body(self) -> QRectF:
        return QRectF(0, 0, self.w, self.h)

    def radius(self) -> float:
        return self.h / 2 if self.node.kind in ("split", "merge") else 8.0

    def has_port(self) -> bool:
        return self.view.editable and not self.node.is_slot

    def port_center(self) -> QPointF:
        return QPointF(self.w / 2, self.h)

    def port_hit(self, local: QPointF) -> bool:
        if not self.has_port():
            return False
        d = local - self.port_center()
        return d.x() * d.x() + d.y() * d.y() <= (PORT_R + 5) ** 2

    def row_at(self, local: QPointF) -> str:
        if self.node.kind != "choice" or local.y() < CHOICE_HEAD_H:
            return ""
        i = int((local.y() - CHOICE_HEAD_H - 3) // CHOICE_ROW_H)
        return self.rows[i][0] if 0 <= i < len(self.rows) else ""

    def row_ids(self) -> List[str]:
        return [cid for cid, _t, _i in self.rows]

    def set_width(self, w: float):
        """Take the lane's column width (layout decides it after measuring every node)."""
        if abs(w - self.w) > 0.01:
            self.prepareGeometryChange()
            self.w = float(w)

    def boundingRect(self) -> QRectF:
        return QRectF(-4, -4, self.w + 8, self.h + 8 + PORT_R * 2)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        path.addRoundedRect(self.body(), self.radius(), self.radius())
        if self.has_port():
            path.addEllipse(self.port_center(), PORT_R + 5, PORT_R + 5)
        return path

    # ---- state ----------------------------------------------------------

    def set_state(self, done: bool, current: bool, row: str):
        if (done, current, row) != (self.done, self.current, self.current_row):
            self.done, self.current, self.current_row = done, current, row
            self.update()

    def set_target(self, on: bool):
        if on != self.target:
            self.target = on
            self.update()

    def hoverEnterEvent(self, ev):
        self.hover = True
        self.update()

    def hoverMoveEvent(self, ev):
        over = self.port_hit(ev.pos())
        if over != self.port_hover:
            self.port_hover = over
            self.update()

    def hoverLeaveEvent(self, ev):
        self.hover = False
        self.port_hover = False
        self.update()

    # ---- painting -------------------------------------------------------

    def paint(self, p: QPainter, option, widget=None):
        th = self.view.theme
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont(self.view.font())
        p.setFont(font)
        fm = QFontMetricsF(font)
        r = self.body()
        kind = self.node.kind
        rad = self.radius()

        if kind == "slot":
            pen = QPen(th.hl if self.target else th.dim, 2.0 if self.target else 1.2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.setBrush(QBrush(_mix(th.base, th.hl, 0.12)) if self.target else Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), rad, rad)
            p.setPen(th.dim)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.label)
            return

        fill = th.node_hover if self.hover else th.node
        if kind in ("split", "merge"):
            fill = _mix(th.base, self.color, (0.30 if th.dark else 0.18) + (0.08 if self.hover else 0.0))
            border = QPen(self.color, 1.3)
        elif kind == "note":
            fill = _mix(th.base, th.note, (0.13 if th.dark else 0.10) + (0.06 if self.hover else 0.0))
            border = QPen(_mix(th.note, th.base, 0.15), 1.2, Qt.PenStyle.DashLine)
        elif kind == "choice":
            border = QPen(th.choice, 1.3)
        else:
            border = QPen(th.border, 1.0)
        if self.done:
            fill = _mix(fill, th.done, 0.24)

        p.setPen(border)
        p.setBrush(QBrush(fill))
        p.drawRoundedRect(r, rad, rad)

        clip = QPainterPath()
        clip.addRoundedRect(r, rad, rad)
        badge_y = r.height() / 2

        if kind == "action":
            p.save()
            p.setClipPath(clip)
            p.fillRect(QRectF(0, 0, 4, r.height()), self.color)
            p.restore()
            x = 12.0
            if not self.icon.isNull():
                self.icon.paint(p, QRect(int(x), int((r.height() - 16) / 2), 16, 16))
                x += 22
            p.setPen(th.text)
            text_w = r.width() - x - (26 if self.done else 10)
            p.drawText(QRectF(x, 0, text_w, r.height()),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       fm.elidedText(self.label, Qt.TextElideMode.ElideRight, text_w))
        elif kind in ("split", "merge"):
            p.setPen(th.text)
            p.drawText(r.adjusted(12, 0, -12, 0), Qt.AlignmentFlag.AlignCenter,
                       fm.elidedText(self.label, Qt.TextElideMode.ElideRight, r.width() - 24))
        elif kind == "note":
            p.setPen(th.text)
            p.drawText(QRectF(8, 6, 20, 20), Qt.AlignmentFlag.AlignCenter, "📝")
            text_w = r.width() - 40 - (18 if self.done else 0)
            lines = _wrap_lines(fm, self.label, text_w)
            line_h = fm.height()
            top = (r.height() - line_h * len(lines)) / 2
            for i, line in enumerate(lines):
                p.drawText(QRectF(32, top + i * line_h, text_w, line_h),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, line)
            badge_y = 13
        elif kind == "choice":
            head = QRectF(0, 0, r.width(), CHOICE_HEAD_H)
            p.save()
            p.setClipPath(clip)
            p.fillRect(head, _mix(th.base, th.choice, 0.30 if th.dark else 0.20))
            for i, (cid, _text, _icon) in enumerate(self.rows):
                if cid == self.current_row:
                    p.fillRect(QRectF(0, CHOICE_HEAD_H + 3 + i * CHOICE_ROW_H, r.width(), CHOICE_ROW_H),
                               _mix(fill, th.hl, 0.35))
            p.restore()
            bold = QFont(font)
            bold.setBold(True)
            p.setFont(bold)
            p.setPen(th.text)
            head_w = r.width() - (34 if self.done else 18)
            p.drawText(QRectF(10, 0, head_w, CHOICE_HEAD_H),
                       Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                       QFontMetricsF(bold).elidedText(self.label, Qt.TextElideMode.ElideRight, head_w))
            p.setFont(font)
            if not self.rows:
                p.setPen(th.dim)
                p.drawText(QRectF(10, CHOICE_HEAD_H + 3, r.width() - 20, CHOICE_ROW_H),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           "Drop tools here as options" if self.view.editable else "No options yet")
            for i, (_cid, text, icon) in enumerate(self.rows):
                top = CHOICE_HEAD_H + 3 + i * CHOICE_ROW_H
                x = 14.0
                if not icon.isNull():
                    icon.paint(p, QRect(int(x), int(top + (CHOICE_ROW_H - 16) / 2), 16, 16))
                x += 22
                p.setPen(th.text)
                p.drawText(QRectF(x, top, r.width() - x - 8, CHOICE_ROW_H),
                           Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                           fm.elidedText(text, Qt.TextElideMode.ElideRight, r.width() - x - 8))
            badge_y = CHOICE_HEAD_H / 2

        if self.done:
            c = QPointF(r.width() - 14, badge_y)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(th.done))
            p.drawEllipse(c, 8, 8)
            p.setPen(QPen(QColor(255, 255, 255), 1.8, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPolyline(QPolygonF([c + QPointF(-3.5, 0.2), c + QPointF(-1.0, 2.8), c + QPointF(3.8, -2.6)]))

        if self.current or self.current_row:
            p.setPen(QPen(th.hl, 2.4 if self.current else 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-2, -2, 2, 2), rad + 2, rad + 2)
        if self.target:
            p.setPen(QPen(th.hl, 2.0, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(-4, -4, 4, 4), rad + 4, rad + 4)

        if self.has_port():
            c = self.port_center()
            lit = self.hover or self.port_hover
            p.setPen(QPen(self.color, 1.4))
            p.setBrush(QBrush(self.color if lit else th.base))
            p.drawEllipse(c, PORT_R + (1.5 if self.port_hover else 0.0), PORT_R + (1.5 if self.port_hover else 0.0))


class _EdgeItem(QGraphicsPathItem):
    """Lane order (thin, lane color) or a branch/join link (thicker, selectable)."""

    def __init__(self, edge: GraphEdge, a: QPointF, b: QPointF, color: QColor,
                 view: "WorkflowGraphView", tip: str = "", avoid: Optional[List[QRectF]] = None,
                 sideways: bool = False):
        super().__init__()
        self.edge = edge
        self.view = view
        self.color = color
        self.is_link = edge.link is not None
        self.selected = False
        self.hover = False
        self.end = b
        path = QPainterPath(a)
        if sideways:
            # Side to side: a link into a step that was dragged up level with,
            # or above, the step that feeds it.
            sign = 1.0 if b.x() >= a.x() else -1.0
            dx = max(30.0, abs(b.x() - a.x()) * 0.5)
            path.cubicTo(QPointF(a.x() + sign * dx, a.y()), QPointF(b.x() - sign * dx, b.y()), b)
        elif abs(a.x() - b.x()) < 0.5:
            path.lineTo(b)
        else:
            dy = max(30.0, (b.y() - a.y()) * 0.5)
            path.cubicTo(QPointF(a.x(), a.y() + dy), QPointF(b.x(), b.y() - dy), b)
        self.setPath(path)
        self.setZValue(0 if self.is_link else -1)
        self.setAcceptHoverEvents(self.is_link)
        if tip:
            self.setToolTip(tip)

        self._label_rect = QRectF()
        if edge.label:
            small = QFont(view.font())
            small.setPointSizeF(max(6.0, small.pointSizeF() - 1.0))
            self._label_font = small
            fm = QFontMetricsF(small)
            w = fm.horizontalAdvance(edge.label) + 12
            h = fm.height() + 4
            # Middle of the curve unless a step box sits there; then slide along it.
            for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8, 0.5):
                mid = path.pointAtPercent(t)
                rect = QRectF(mid.x() - w / 2, mid.y() - h / 2, w, h)
                if not any(rect.intersects(r) for r in (avoid or [])):
                    break
            self._label_rect = rect

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(12 if self.is_link else 4)
        return stroker.createStroke(self.path())

    def boundingRect(self) -> QRectF:
        r = self.path().boundingRect().adjusted(-10, -10, 10, 10)
        return r.united(self._label_rect.adjusted(-2, -2, 2, 2)) if not self._label_rect.isNull() else r

    def set_selected(self, on: bool):
        if on != self.selected:
            self.selected = on
            self.update()

    def hoverEnterEvent(self, ev):
        self.hover = True
        self.update()

    def hoverLeaveEvent(self, ev):
        self.hover = False
        self.update()

    def paint(self, p: QPainter, option, widget=None):
        th = self.view.theme
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        col = QColor(th.hl if self.selected else self.color)
        if not self.is_link:
            col.setAlpha(170)
        width = 2.8 if (self.selected or self.hover) else (2.0 if self.is_link else 1.5)
        p.setPen(QPen(col, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = self.path()
        p.drawPath(path)

        tip = self.end
        prev = path.pointAtPercent(0.96)
        dx, dy = tip.x() - prev.x(), tip.y() - prev.y()
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ux, uy = dx / norm, dy / norm
        size = 8.0 if self.is_link else 6.5
        back = QPointF(tip.x() - ux * size, tip.y() - uy * size)
        perp = QPointF(-uy * size * 0.55, ux * size * 0.55)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(col))
        p.drawPolygon(QPolygonF([tip, back + perp, back - perp]))

        if not self._label_rect.isNull():
            p.setFont(self._label_font)
            p.setPen(QPen(col, 1.0))
            p.setBrush(QBrush(th.base))
            r = self._label_rect
            p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            p.setPen(th.text)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.edge.label)


class _LaneItem(QGraphicsItem):
    """A lane's column band and its header."""

    def __init__(self, lane: str, x: float, w: float, height: float, color: QColor,
                 view: "WorkflowGraphView"):
        super().__init__()
        self.lane = lane
        self.color = color
        self.view = view
        self.target = False
        self.rect = QRectF(x, MARGIN, w, height - 2 * MARGIN)
        self.header = QRectF(x + 8, MARGIN + 6, w - 16, HEADER_H - 6)
        self.setZValue(-5)
        tip = f"Lane <b>{html.escape(lane)}</b>"
        if view.editable:
            tip += "<br>Double-click the name to rename. Right-click for lane commands."
        self.setToolTip(tip)

    def boundingRect(self) -> QRectF:
        return self.rect.adjusted(-2, -2, 2, 2)

    def set_target(self, on: bool):
        if on != self.target:
            self.target = on
            self.update()

    def paint(self, p: QPainter, option, widget=None):
        th = self.view.theme
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        band = _mix(th.base, self.color, (0.07 if th.dark else 0.06) + (0.08 if self.target else 0.0))
        p.setPen(QPen(th.hl, 1.6, Qt.PenStyle.DashLine) if self.target
                 else QPen(_mix(th.base, self.color, 0.22), 1.0))
        p.setBrush(QBrush(band))
        p.drawRoundedRect(self.rect, 10, 10)

        h = self.header
        p.setPen(QPen(self.color, 1.2))
        p.setBrush(QBrush(_mix(th.base, self.color, 0.28 if th.dark else 0.20)))
        p.drawRoundedRect(h, h.height() / 2, h.height() / 2)
        bold = QFont(self.view.font())
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(th.text)
        p.drawText(h.adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignCenter,
                   QFontMetricsF(bold).elidedText(self.lane, Qt.TextElideMode.ElideRight, h.width() - 20))


class _NewLaneItem(QGraphicsItem):
    """Dashed column after the last lane: drop tools or a link here to start a lane."""

    def __init__(self, x: float, height: float, view: "WorkflowGraphView"):
        super().__init__()
        self.view = view
        self.target = False
        self.rect = QRectF(x, MARGIN, NEW_LANE_W, height - 2 * MARGIN)
        self.setZValue(-5)
        self.setToolTip("Drop a tool here, or drag a step's handle here, to start a new lane.<br>"
                        "Double-click to add an empty lane.")

    def boundingRect(self) -> QRectF:
        return self.rect.adjusted(-2, -2, 2, 2)

    def set_target(self, on: bool):
        if on != self.target:
            self.target = on
            self.update()

    def paint(self, p: QPainter, option, widget=None):
        th = self.view.theme
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        col = th.hl if self.target else th.dim
        p.setPen(QPen(col, 1.6 if self.target else 1.1, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(_mix(th.base, th.hl, 0.10)) if self.target else Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(self.rect, 10, 10)
        p.setPen(col)
        p.drawText(QRectF(self.rect.left(), MARGIN + 6, self.rect.width(), HEADER_H - 6),
                   Qt.AlignmentFlag.AlignCenter, "+ New lane")


# -----------------------------------------------------------------------------
# View
# -----------------------------------------------------------------------------

class WorkflowGraphView(QGraphicsView):
    """Interactive graph. All state changes go through `host` (a _WorkflowGraphHost)."""

    zoomChanged = pyqtSignal(float)

    def __init__(self, host, editable: bool, parent=None):
        super().__init__(parent)
        self.host = host
        self.editable = bool(editable)
        self.theme = _Theme(self)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setAcceptDrops(self.editable)
        self.viewport().setAcceptDrops(self.editable)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._graph: Optional[WorkflowGraph] = None
        self._wf: Optional[WorkflowDefinition] = None
        self._nodes: Dict[str, _NodeItem] = {}
        self._lanes: Dict[str, _LaneItem] = {}
        self._edges: List[_EdgeItem] = []
        self._new_lane: Optional[_NewLaneItem] = None
        self._row_owner: Dict[str, str] = {}
        self._targets: list = []
        self._marker: Optional[QGraphicsPathItem] = None
        self._ghost: Optional[QGraphicsPathItem] = None
        self._temp_link: Optional[QGraphicsPathItem] = None
        self._selected_link: Optional[_EdgeItem] = None

        self._mode = ""            # "", press, move, link, pan
        self._press_pos = QPointF()
        self._press_item: Optional[_NodeItem] = None
        self._move_offset = QPointF()
        self._move_natural = 0
        self._link_src = ""
        self._pan_last = QPointF()
        self._last_current = ""
        self._zoom = 1.0
        self._fit_pending = True
        self._auto_fit = False

        self._hint = QLabel(self.viewport())
        self._hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._hint.hide()
        self._hint_timer = QTimer(self)
        self._hint_timer.setSingleShot(True)
        self._hint_timer.timeout.connect(self._hint.hide)

    # ---- building -------------------------------------------------------

    def rebuild(self):
        self._cancel_interaction(refresh=False)
        self.theme = _Theme(self)
        wf = self.host.graph_workflow()
        catalog = self.host.graph_catalog() or {}
        self._wf = wf

        scene = self.scene()
        scene.clear()
        self._nodes.clear()
        self._lanes.clear()
        self._edges.clear()
        self._targets.clear()
        self._row_owner.clear()
        self._new_lane = None
        self._marker = None
        self._ghost = None
        self._temp_link = None
        self._selected_link = None

        g = build_workflow_graph(wf)
        lane_index = {lane: i for i, lane in enumerate(g.lanes)}
        items = {key: _NodeItem(n, self, self.theme.lane(lane_index[n.lane]), catalog)
                 for key, n in g.nodes.items()}
        place_workflow_graph(g, {k: (it.w, it.h) for k, it in items.items()},
                             reserve_new_lane=self.editable)
        self._graph = g

        for lane in g.lanes:
            x, w = g.columns[lane]
            item = _LaneItem(lane, x, w, g.height, self.theme.lane(lane_index[lane]), self)
            scene.addItem(item)
            self._lanes[lane] = item
        if self.editable:
            self._new_lane = _NewLaneItem(g.new_lane_x, g.height, self)
            scene.addItem(self._new_lane)

        for key, item in items.items():
            n = g.nodes[key]
            item.set_width(n.w)
            item.setPos(n.x, n.y)
            scene.addItem(item)
            self._nodes[key] = item
            for cid in item.row_ids():
                self._row_owner[cid] = key

        self._add_edges(g, catalog)
        scene.setSceneRect(QRectF(0, 0, g.width, g.height))
        self._last_current = ""
        self.restyle()
        if self._fit_pending and self.isVisible():
            self.fit()

    def _add_edges(self, g: WorkflowGraph, catalog):
        incoming: Dict[str, List[GraphEdge]] = {}
        for e in g.edges:
            incoming.setdefault(e.dst, []).append(e)
        lane_index = {lane: i for i, lane in enumerate(g.lanes)}
        boxes = [QRectF(n.x - 4, n.y - 4, n.w + 8, n.h + 8) for n in g.nodes.values()]
        for dst, edges in incoming.items():
            d = g.nodes[dst]
            side = [e for e in edges if self._enters_side(g, e)]
            top = [e for e in edges if not any(e is x for x in side)]
            placed = []
            for e, spread in self._arrival_offsets(g, d, top):
                s = g.nodes[e.src]
                placed.append((e, QPointF(s.x + s.w / 2, s.y + s.h),
                               QPointF(d.x + d.w / 2 + spread, d.y), False))
            placed += [(e, a, b, True) for e, a, b in self._side_points(g, d, side)]
            for e, a, b, sideways in placed:
                s = g.nodes[e.src]
                color = self.theme.lane(lane_index[s.lane])
                tip = ""
                if e.link is not None:
                    src_t = format_workflow_step_label(s.step, catalog) if s.step else s.key
                    if e.link.dst:
                        dst_t = format_workflow_step_label(d.step, catalog) if d.step else d.key
                        tip = f"Join: {html.escape(src_t)} → {html.escape(dst_t)} ({html.escape(d.lane)})"
                    else:
                        tip = f"Branch: {html.escape(src_t)} → lane {html.escape(d.lane)}"
                    if e.label:
                        tip += f"<br><i>Carries:</i> {html.escape(e.label)}"
                    if self.editable:
                        tip += "<br>Click to select, Delete to remove."
                item = _EdgeItem(e, a, b, color, self, tip, avoid=boxes, sideways=sideways)
                self.scene().addItem(item)
                self._edges.append(item)

    @staticmethod
    def _enters_side(g: WorkflowGraph, e: GraphEdge) -> bool:
        """A link whose target no longer sits below its source (it was dragged up)."""
        if e.link is None:
            return False
        s, d = g.nodes[e.src], g.nodes[e.dst]
        return d.y < s.y + s.h + 4

    @staticmethod
    def _side_points(g: WorkflowGraph, d: GraphNode, edges: List[GraphEdge]):
        """(edge, start, end) leaving the source's facing side and entering d's facing side."""
        out = []
        center = d.x + d.w / 2
        for from_left in (True, False):
            group = [e for e in edges if (g.nodes[e.src].x + g.nodes[e.src].w / 2 < center) == from_left]
            group.sort(key=lambda e: g.nodes[e.src].y)
            n = len(group)
            step = min(8.0, d.h / (n + 1))
            for i, e in enumerate(group):
                s = g.nodes[e.src]
                off = (i - (n - 1) / 2) * step
                a = QPointF(s.x + s.w if from_left else s.x, s.y + s.h / 2)
                b = QPointF(d.x if from_left else d.x + d.w, d.y + d.h / 2 + off)
                out.append((e, a, b))
        return out

    @staticmethod
    def _arrival_offsets(g: WorkflowGraph, d: GraphNode, edges: List[GraphEdge]):
        """Where each edge meets the top of d: lane order dead centre, links fanned to their side."""
        center = d.x + d.w / 2

        def src_x(e):
            s = g.nodes[e.src]
            return s.x + s.w / 2

        seq = [e for e in edges if e.link is None]
        links = [e for e in edges if e.link is not None]
        if not seq:
            links.sort(key=src_x)
            n = len(links)
            return [(e, 0.0 if n == 1 else (i / (n - 1) - 0.5) * 0.6 * d.w) for i, e in enumerate(links)]
        left = sorted((e for e in links if src_x(e) < center), key=src_x, reverse=True)
        right = sorted((e for e in links if src_x(e) >= center), key=src_x)
        step = min(20.0, 0.4 * d.w / max(1, len(left), len(right)))
        out = [(e, 0.0) for e in seq]
        out += [(e, -step * (i + 1)) for i, e in enumerate(left)]
        out += [(e, step * (i + 1)) for i, e in enumerate(right)]
        return out

    def restyle(self):
        if not self._nodes:
            return
        done = self.host.graph_completed_ids()
        cur = self.host.graph_current_id()
        owner = self._row_owner.get(cur, cur)
        for key, item in self._nodes.items():
            item.set_state(done=key in done, current=(key == cur),
                           row=cur if (owner == key and cur != key) else "")
        if cur != self._last_current:
            self._last_current = cur
            item = self._nodes.get(owner)
            if item is not None and self.isVisible() and not self._mode:
                self.ensureVisible(item.sceneBoundingRect(), 30, 30)

    # ---- zoom -----------------------------------------------------------

    def request_fit(self):
        self._fit_pending = True
        if self.isVisible() and self._graph is not None:
            QTimer.singleShot(0, self._fit_if_pending)

    def _fit_if_pending(self):
        if self._fit_pending and self._graph is not None:
            self.fit()

    def _fit_zoom(self) -> bool:
        rect = self.scene().sceneRect()
        vp = self.viewport().rect()
        if rect.isEmpty() or vp.width() < 40 or vp.height() < 40:
            return False
        whole = min(1.0, (vp.width() - 8) / rect.width(), (vp.height() - 8) / rect.height())
        if whole >= FIT_MIN:
            s = whole
        else:
            # It cannot all show at a readable size: keep the text readable,
            # fit the width if possible, and scroll for the rest.
            s = max(FIT_MIN, min(1.0, (vp.width() - 8) / rect.width()))
        self.set_zoom(s)
        self._auto_fit = True     # keep fitting on resize until the user zooms by hand
        return True

    def fit(self):
        if not self._fit_zoom():
            self._fit_pending = True
            return
        self._fit_pending = False
        rect = self.scene().sceneRect()
        self.ensureVisible(QRectF(rect.left(), rect.top(), 1, 1), 0, 0)
        cur = self._nodes.get(self._row_owner.get(self._last_current, self._last_current))
        if cur is not None:
            self.ensureVisible(cur.sceneBoundingRect(), 30, 30)

    def set_zoom(self, s: float):
        s = _clamp(s, ZOOM_MIN, ZOOM_MAX)
        self.setTransform(QTransform.fromScale(s, s))
        self._zoom = s
        self.zoomChanged.emit(s)

    def zoom_by(self, factor: float, under_mouse: bool = False):
        self._auto_fit = False
        target = _clamp(self._zoom * factor, ZOOM_MIN, ZOOM_MAX)
        if abs(target - self._zoom) < 1e-6:
            return
        if not under_mouse:
            self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(target / self._zoom, target / self._zoom)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = target
        self.zoomChanged.emit(target)

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            steps = ev.angleDelta().y() / 120.0
            if steps:
                self.zoom_by(1.15 ** steps, under_mouse=True)
            ev.accept()
            return
        super().wheelEvent(ev)

    # ---- hints ----------------------------------------------------------

    def show_hint(self, text: str, sticky: bool = False):
        if not text:
            self._hint.hide()
            return
        th = self.theme
        bg = _mix(th.base, th.text, 0.10)
        self._hint.setStyleSheet(
            f"background: rgba({bg.red()},{bg.green()},{bg.blue()},235); color: {th.text.name()};"
            f"border: 1px solid {th.hl.name()}; border-radius: 6px; padding: 4px 9px;"
        )
        self._hint.setText(text)
        self._hint.adjustSize()
        self._hint.move(10, max(4, self.viewport().height() - self._hint.height() - 10))
        self._hint.show()
        self._hint.raise_()
        if sticky:
            self._hint_timer.stop()
        else:
            self._hint_timer.start(4500)

    def _hide_sticky_hint(self):
        if not self._hint_timer.isActive():
            self._hint.hide()

    # ---- hit testing ----------------------------------------------------

    def _hit(self, sp: QPointF, skip=None):
        """(kind, item, row) of what sits under scene point sp."""
        for it in self.scene().items(sp):
            if it is skip:
                continue
            if isinstance(it, _NodeItem):
                local = it.mapFromScene(sp)
                if it.port_hit(local):
                    return "port", it, ""
                return "node", it, it.row_at(local)
            if isinstance(it, _EdgeItem):
                if it.is_link:
                    return "link", it, ""
                continue
            if isinstance(it, _LaneItem):
                return ("header" if it.header.contains(sp) else "lane"), it, ""
            if isinstance(it, _NewLaneItem):
                return "newlane", it, ""
        return "canvas", None, ""

    def _lane_at_x(self, x: float) -> Optional[str]:
        """Lane whose column holds x; "" for the new-lane column."""
        g = self._graph
        if g is None or not g.lanes:
            return None
        for lane in g.lanes:
            cx, cw = g.columns[lane]
            if x < cx + cw + COL_GAP / 2:
                return lane
        if self.editable and x >= g.new_lane_x - COL_GAP / 2:
            return ""
        return g.lanes[-1]

    def _insert_slot(self, lane: str, y: float, exclude: str = "") -> Tuple[int, float]:
        """(index among the lane's steps, y of the insertion marker)."""
        g = self._graph
        nodes = [g.nodes[k] for k in g.order.get(lane, [])
                 if k != exclude and not k.startswith(SLOT_PREFIX)]
        index = sum(1 for n in nodes if n.y + n.h / 2 < y)
        if not nodes:
            slot = g.nodes.get(slot_key(lane))
            my = slot.y + slot.h / 2 if slot is not None else MARGIN + HEADER_H + HEADER_GAP
        elif index == 0:
            my = nodes[0].y - ROW_GAP / 2
        elif index == len(nodes):
            my = nodes[-1].y + nodes[-1].h + ROW_GAP / 2
        else:
            my = (nodes[index - 1].y + nodes[index - 1].h + nodes[index].y) / 2
        return index, my

    # ---- feedback -------------------------------------------------------

    def _set_targets(self, items: list):
        for it in self._targets:
            try:
                it.set_target(False)
            except RuntimeError:
                pass
        self._targets = list(items)
        for it in self._targets:
            it.set_target(True)

    def _show_marker(self, lane: str, y: float):
        cx, cw = self._graph.columns[lane]
        path = QPainterPath(QPointF(cx + 10, y))
        path.lineTo(QPointF(cx + cw - 10, y))
        path.addEllipse(QPointF(cx + 10, y), 3.5, 3.5)
        path.addEllipse(QPointF(cx + cw - 10, y), 3.5, 3.5)
        if self._marker is None:
            self._marker = QGraphicsPathItem()
            self._marker.setZValue(20)
            self._marker.setPen(QPen(self.theme.hl, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            self._marker.setBrush(QBrush(self.theme.hl))
            self.scene().addItem(self._marker)
        self._marker.setPath(path)
        self._marker.show()

    def _hide_marker(self):
        if self._marker is not None:
            self._marker.hide()

    def _clear_feedback(self):
        self._set_targets([])
        self._hide_marker()
        if self._ghost is not None:
            self._ghost.hide()
        self._hide_sticky_hint()

    def _autoscroll(self, pos: QPointF):
        edge = 24
        vp = self.viewport().rect()
        dx = dy = 0
        if pos.x() < edge:
            dx = -12
        elif pos.x() > vp.width() - edge:
            dx = 12
        if pos.y() < edge:
            dy = -12
        elif pos.y() > vp.height() - edge:
            dy = 12
        if dx:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + dx)
        if dy:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + dy)

    def _select_link(self, item: Optional[_EdgeItem]):
        if self._selected_link is not None and self._selected_link is not item:
            try:
                self._selected_link.set_selected(False)
            except RuntimeError:
                pass
        self._selected_link = item
        if item is not None:
            item.set_selected(True)

    # ---- tool drops -----------------------------------------------------

    def _drop_target(self, sp: QPointF):
        """(lane, index, choice_id) for a tool dropped at sp; lane "" starts a new lane."""
        lane = self._lane_at_x(sp.x())
        if lane is None:
            return None
        if lane == "":
            return "", 0, ""
        for it in self.scene().items(sp):
            if isinstance(it, _NodeItem) and it.node.kind == "choice" and it.node.lane == lane:
                return lane, 0, it.node.key
        index, _y = self._insert_slot(lane, sp.y())
        return lane, index, ""

    def _drop_feedback(self, pos: QPointF):
        self._autoscroll(pos)
        sp = self.mapToScene(pos.toPoint())
        target = self._drop_target(sp)
        if target is None:
            self._clear_feedback()
            return
        lane, _index, choice = target
        if choice:
            self._set_targets([self._nodes[choice]])
            self._hide_marker()
            self.show_hint("Add as an option of this choice", sticky=True)
        elif lane == "":
            self._set_targets([self._new_lane] if self._new_lane is not None else [])
            self._hide_marker()
            self.show_hint("Start a new lane", sticky=True)
        else:
            self._set_targets([])
            _i, y = self._insert_slot(lane, sp.y())
            self._show_marker(lane, y)
            self._hide_sticky_hint()

    def dragEnterEvent(self, ev):
        if self.editable and has_tool_payload(ev.mimeData()):
            ev.acceptProposedAction()
            self._drop_feedback(ev.position())
        else:
            ev.ignore()

    def dragMoveEvent(self, ev):
        if self.editable and has_tool_payload(ev.mimeData()):
            ev.acceptProposedAction()
            self._drop_feedback(ev.position())
        else:
            ev.ignore()

    def dragLeaveEvent(self, ev):
        self._clear_feedback()
        ev.accept()

    def dropEvent(self, ev):
        if not self.editable:
            ev.ignore()
            return
        payloads = tool_payloads(ev.mimeData())
        target = self._drop_target(self.mapToScene(ev.position().toPoint()))
        self._clear_feedback()
        if not payloads or target is None:
            ev.ignore()
            return
        ev.acceptProposedAction()
        lane, index, choice = target
        # Let the drag finish before the host rebuilds lane lists and this scene.
        QTimer.singleShot(0, lambda: self.host.graph_insert(payloads, lane, index, choice))

    # ---- links ----------------------------------------------------------

    def _link_target(self, sp: QPointF):
        """(kind, value, text) for a link dropped at sp. kind: step, lane, new, bad or ""."""
        src = self._graph.nodes.get(self._link_src)
        if src is None:
            return "", None, ""
        kind, item, _row = self._hit(sp)
        if kind in ("node", "port"):
            n = item.node
            if n.key == src.key:
                return "", None, ""
            if n.lane == src.lane:
                return "bad", None, "Same lane: the lane order already connects these steps."
            if n.is_slot:
                kind, value, text = "lane", n.lane, f"Branch into lane '{n.lane}'"
            else:
                kind, value, text = "step", n.key, f"Join into '{item.label}' in lane '{n.lane}'"
        elif kind in ("lane", "header"):
            if item.lane == src.lane:
                return "bad", None, "Same lane: the lane order already connects these steps."
            kind, value, text = "lane", item.lane, f"Branch into the start of lane '{item.lane}'"
        elif kind == "newlane":
            return "new", None, "Branch into a new lane"
        else:
            return "", None, "Drop on a step or a lane in another column"

        wf = self._wf
        if wf is not None:
            new = WorkflowLink(src=src.key, dst=value if kind == "step" else "",
                               dst_lane=value if kind == "lane" else "")
            dropped: list = []
            normalize_workflow_links(list(wf.links or []) + [new], wf.steps, wf.lanes, dropped)
            for link, reason in dropped:
                if link is new:
                    return "bad", None, LINK_DROP_REASONS.get(reason, "That link is not allowed.")
        return kind, value, text

    def _begin_link(self, item: _NodeItem):
        self._mode = "link"
        self._link_src = item.node.key
        self._temp_link = QGraphicsPathItem()
        self._temp_link.setZValue(30)
        self._temp_link.setPen(QPen(self.theme.hl, 2.2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        self.scene().addItem(self._temp_link)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _drag_link(self, pos: QPointF):
        self._autoscroll(pos)
        sp = self.mapToScene(pos.toPoint())
        src = self._nodes.get(self._link_src)
        if src is None or self._temp_link is None:
            return
        a = src.mapToScene(src.port_center())
        dy = max(30.0, abs(sp.y() - a.y()) * 0.5)
        path = QPainterPath(a)
        path.cubicTo(QPointF(a.x(), a.y() + dy), QPointF(sp.x(), sp.y() - dy), sp)
        self._temp_link.setPath(path)

        kind, value, text = self._link_target(sp)
        targets = []
        if kind == "step":
            targets = [self._nodes[value]]
        elif kind == "lane":
            slot = self._nodes.get(slot_key(value))
            targets = [slot] if slot is not None else [self._lanes[value]]
        elif kind == "new" and self._new_lane is not None:
            targets = [self._new_lane]
        self._set_targets(targets)
        self.show_hint(text, sticky=True)

    def _finish_link(self, pos: QPointF):
        sp = self.mapToScene(pos.toPoint())
        kind, value, text = self._link_target(sp)
        src = self._link_src
        self._end_link()
        if kind == "step":
            self._report(self.host.graph_link(src, dst=value))
        elif kind == "lane":
            self._report(self.host.graph_link(src, dst_lane=value))
        elif kind == "new":
            QTimer.singleShot(0, lambda: self.host.graph_branch_new_lane(src, at_end=True))
        elif kind == "bad":
            self.show_hint(text)

    def _report(self, error: str):
        if error:
            self.show_hint(error)

    def _end_link(self):
        if self._temp_link is not None:
            self.scene().removeItem(self._temp_link)
            self._temp_link = None
        self._link_src = ""
        self._mode = ""
        self._clear_feedback()
        self.viewport().unsetCursor()

    # ---- node moves -----------------------------------------------------

    def _begin_move(self, sp: QPointF):
        item = self._press_item
        self._mode = "move"
        self._move_offset = sp - item.pos()
        self._move_natural = natural_rank(self._wf, item.node.key) if self._wf is not None else 0
        item.setZValue(25)
        item.setOpacity(0.8)
        for e in self._edges:
            if item.node.key in (e.edge.src, e.edge.dst):
                e.hide()
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def _move_target(self, sp: QPointF):
        """What dropping the dragged step here would do.

        ("newlane",) | ("reorder", lane, index, marker_y) | ("row", lane, row).
        Within its own lane, a drop that would not change the order sets the
        step's height instead: anywhere between the steps above and below it.
        """
        item = self._press_item
        key = item.node.key
        lane = self._lane_at_x(sp.x())
        if lane is None:
            return None
        if lane == "":
            return ("newlane",)
        cy = sp.y() - self._move_offset.y() + item.h / 2      # centre of the dragged box
        index, my = self._insert_slot(lane, cy, exclude=key)
        g = self._graph
        if lane == item.node.lane:
            steps = [k for k in g.order[lane] if not k.startswith(SLOT_PREFIX)]
            pos = steps.index(key)
            if index == pos:
                lo = g.nodes[steps[pos - 1]].rank + 1 if pos > 0 else 0
                hi = g.nodes[steps[pos + 1]].rank - 1 if pos + 1 < len(steps) else len(g.row_top)
                return ("row", lane, max(lo, min(row_at_y(g, cy), max(lo, hi))))
        return ("reorder", lane, index, my)

    def _show_ghost(self, lane: str, row: int):
        """Dashed outline where the dragged step will land."""
        g = self._graph
        item = self._press_item
        cx, cw = g.columns[lane]
        w, h = item.w, item.h
        y = row_top_y(g, row) + (row_height(g, row) - h) / 2
        path = QPainterPath()
        r = item.radius()
        path.addRoundedRect(QRectF(cx + (cw - w) / 2, y, w, h), r, r)
        if self._ghost is None:
            self._ghost = QGraphicsPathItem()
            self._ghost.setZValue(20)
            pen = QPen(self.theme.hl, 2.0, Qt.PenStyle.DashLine)
            self._ghost.setPen(pen)
            fill = QColor(self.theme.hl)
            fill.setAlpha(40)
            self._ghost.setBrush(QBrush(fill))
            self.scene().addItem(self._ghost)
        self._ghost.setPath(path)
        self._ghost.show()

    def _drag_move(self, pos: QPointF):
        self._autoscroll(pos)
        sp = self.mapToScene(pos.toPoint())
        item = self._press_item
        item.setPos(sp - self._move_offset)
        target = self._move_target(sp)
        if target is None:
            return
        if target[0] == "newlane":
            self._set_targets([self._new_lane] if self._new_lane is not None else [])
            self._hide_marker()
            if self._ghost is not None:
                self._ghost.hide()
            self.show_hint("Move into a new lane", sticky=True)
        elif target[0] == "row":
            self._set_targets([])
            self._hide_marker()
            _kind, lane, row = target
            self._show_ghost(lane, row)
            if row == self._move_natural:
                self.show_hint("Back to its automatic height", sticky=True)
            else:
                self.show_hint("Keep it at this height · right-click › Auto Position undoes it", sticky=True)
        else:
            self._set_targets([])
            if self._ghost is not None:
                self._ghost.hide()
            _kind, lane, _index, y = target
            self._show_marker(lane, y)
            self._hide_sticky_hint()

    def _finish_move(self, pos: QPointF):
        sp = self.mapToScene(pos.toPoint())
        key = self._press_item.node.key
        target = self._move_target(sp)
        self._mode = ""
        self._press_item = None
        self._clear_feedback()
        self.viewport().unsetCursor()
        if target is not None:
            if target[0] == "newlane":
                self.host.graph_move(key, "", 0)
            elif target[0] == "row":
                row = target[2]
                self.host.graph_set_row(key, None if row == self._move_natural else row)
            else:
                self.host.graph_move(key, target[1], target[2])
        self.host.graph_refresh()   # also puts the node back when nothing moved

    def _cancel_interaction(self, refresh: bool = True):
        mode = self._mode
        if mode == "link":
            self._end_link()
        self._mode = ""
        self._press_item = None
        self._clear_feedback()
        self.viewport().unsetCursor()
        if mode == "move" and refresh:
            self.host.graph_refresh()

    # ---- mouse ----------------------------------------------------------

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position()
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._mode = "pan"
            self._pan_last = pos
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        sp = self.mapToScene(pos.toPoint())
        kind, item, row = self._hit(sp)
        if kind == "port":
            self._select_link(None)
            self._begin_link(item)
            return
        if kind == "node":
            self._select_link(None)
            if not item.node.is_slot:
                sel = row or item.node.key
                self._last_current = sel      # selecting from here: do not scroll
                self.host.graph_select(sel)
                if self.editable:
                    self._mode = "press"
                    self._press_pos = pos
                    self._press_item = item
            return
        if kind == "link":
            self._select_link(item)
            return
        self._select_link(None)
        self._mode = "pan"
        self._pan_last = pos
        self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        mode = self._mode
        if mode == "pan":
            d = pos - self._pan_last
            self._pan_last = pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(round(d.x())))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(round(d.y())))
            return
        if mode == "press":
            if (pos - self._press_pos).manhattanLength() >= QApplication.startDragDistance():
                self._begin_move(self.mapToScene(self._press_pos.toPoint()))
                self._drag_move(pos)
            return
        if mode == "move":
            self._drag_move(pos)
            return
        if mode == "link":
            self._drag_link(pos)
            return
        super().mouseMoveEvent(ev)
        if self.editable:
            kind, _item, _row = self._hit(self.mapToScene(pos.toPoint()))
            if kind == "port":
                self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.viewport().unsetCursor()

    def mouseReleaseEvent(self, ev):
        mode = self._mode
        pos = ev.position()
        if mode == "move":
            self._finish_move(pos)
        elif mode == "link":
            self._finish_link(pos)
        else:
            self._mode = ""
            self._press_item = None
            self.viewport().unsetCursor()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        self._mode = ""
        self._press_item = None
        sp = self.mapToScene(ev.position().toPoint())
        kind, item, row = self._hit(sp)
        if kind == "node" and not item.node.is_slot:
            target = row or item.node.key
            QTimer.singleShot(0, lambda: self.host.graph_open(target))
        elif self.editable and kind == "header":
            lane = item.lane
            QTimer.singleShot(0, lambda: self.host.graph_rename_lane(lane))
        elif self.editable and kind == "newlane":
            QTimer.singleShot(0, self.host.graph_add_lane)

    def contextMenuEvent(self, ev):
        if self._mode:
            return
        sp = self.mapToScene(ev.pos())
        kind, item, row = self._hit(sp)
        menu = QMenu(self)
        if kind in ("node", "port") and not item.node.is_slot:
            self._last_current = row or item.node.key
            self.host.graph_node_menu(menu, row or item.node.key)
        elif kind in ("node", "lane", "header"):
            self.host.graph_lane_menu(menu, item.node.lane if kind == "node" else item.lane)
        elif kind == "link":
            self._select_link(item)
            self.host.graph_link_menu(menu, item.edge.link)
        else:
            self.host.graph_canvas_menu(menu)
        if not menu.isEmpty():
            menu.addSeparator()
        menu.addAction("Fit to Window", self.fit)
        menu.exec(ev.globalPos())

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key.Key_Escape and self._mode:
            self._cancel_interaction()
            return
        if self.editable and key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selected_link is not None:
                link = self._selected_link.edge.link
                self._select_link(None)
                self.host.graph_unlink(link)
                return
            cur = self.host.graph_current_id()
            if cur:
                self.host.graph_delete(cur)
                return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            cur = self.host.graph_current_id()
            if cur:
                self.host.graph_open(cur)
            return
        if key == Qt.Key.Key_F:
            self.fit()
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.15)
            return
        if key == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.15)
            return
        if key == Qt.Key.Key_0 and ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.set_zoom(1.0)
            self._auto_fit = False
            return
        super().keyPressEvent(ev)

    # ---- widget events --------------------------------------------------

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._fit_pending:
            QTimer.singleShot(0, self._fit_if_pending)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._graph is not None:
            if self._fit_pending:
                self.fit()
            elif self._auto_fit:
                self._fit_zoom()
        if self._hint.isVisible():
            self._hint.move(10, max(4, self.viewport().height() - self._hint.height() - 10))

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() in (QEvent.Type.PaletteChange, QEvent.Type.FontChange):
            self.host.graph_refresh()


class WorkflowGraphWindow(QWidget):
    """The graph in its own resizable window.

    Tool palette (editor only) | graph | step details and the step buttons.
    There is one per workflow dialog. It is made on first open, hidden rather
    than destroyed on close, and owned by the dialog, so it goes away with it.
    The palette and buttons let it work on its own when it is maximized over
    the dialog or moved to another monitor.
    """

    def __init__(self, host, editable: bool, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.host = host
        self.editable = bool(editable)
        self._prefix = str(getattr(host, "_GRAPH_SETTINGS_PREFIX", "") or "workflow_graph")
        self._placed = False
        self.setWindowTitle("Workflow Graph")
        self.resize(1400, 900)

        self.view = WorkflowGraphView(host, editable, self)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 8)
        root.setSpacing(6)

        # ---- top bar: workflow name, how-to line, lane button, zoom ----
        bar = QHBoxLayout()
        self.lbl_title = QLabel(self)
        font = self.lbl_title.font()
        font.setBold(True)
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 1.15)
        self.lbl_title.setFont(font)
        bar.addWidget(self.lbl_title)
        bar.addSpacing(14)
        if editable:
            tip = ("Drag a step to reorder it or change lane · drag its ● onto another lane "
                   "to branch, or onto a step to join · right-click for more · Ctrl+wheel zooms")
        else:
            tip = "Click to select · double-click to open · right-click for options · Ctrl+wheel zooms"
        how = QLabel(tip, self)
        how.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        how.setMinimumWidth(40)
        bar.addWidget(how, 1)
        if editable:
            self.btn_add_lane = QPushButton("Add Lane…", self)
            self.btn_add_lane.clicked.connect(lambda _c=False: self.host.graph_add_lane())
            bar.addWidget(self.btn_add_lane)
            bar.addSpacing(8)

        self.btn_out = QToolButton(self)
        self.btn_out.setText("−")
        self.btn_out.setToolTip("Zoom out (−)")
        self.btn_out.setAutoRaise(True)
        self.lbl_zoom = QLabel("100%", self)
        self.lbl_zoom.setMinimumWidth(40)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_in = QToolButton(self)
        self.btn_in.setText("+")
        self.btn_in.setToolTip("Zoom in (+)")
        self.btn_in.setAutoRaise(True)
        self.btn_fit = QToolButton(self)
        self.btn_fit.setText("Fit")
        self.btn_fit.setToolTip("Fit the workflow to the window (F)")
        self.btn_fit.setAutoRaise(True)
        for w in (self.btn_out, self.lbl_zoom, self.btn_in, self.btn_fit):
            bar.addWidget(w)
        root.addLayout(bar)

        self.split = QSplitter(Qt.Orientation.Horizontal, self)
        root.addWidget(self.split, 1)

        # ---- left: tool palette (editor only) ----
        self.edt_search: Optional[QLineEdit] = None
        self.tool_tree = None
        if editable:
            pal = QWidget(self.split)
            pl = QVBoxLayout(pal)
            pl.setContentsMargins(0, 0, 0, 0)
            pl.setSpacing(4)
            self.edt_search = QLineEdit(pal)
            self.edt_search.setPlaceholderText("Search tools...")
            self.edt_search.setClearButtonEnabled(True)
            self.edt_search.textChanged.connect(self._refilter)
            self.tool_tree = WorkflowToolTree(pal)
            self.tool_tree.itemDoubleClicked.connect(self._on_tool_double_clicked)
            pal_tip = QLabel("Drag a tool onto a lane. Double-click adds it after the selected step.", pal)
            pal_tip.setWordWrap(True)
            pal_tip.setForegroundRole(QPalette.ColorRole.PlaceholderText)
            pl.addWidget(self.edt_search)
            pl.addWidget(self.tool_tree, 1)
            pl.addWidget(pal_tip)
            self.split.addWidget(pal)

        # ---- centre: the graph ----
        self.split.addWidget(self.view)

        # ---- right: what the selected step is, and the step buttons ----
        side = QWidget(self.split)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self.lbl_details = QLabel("Select a step to view details.", side)
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.lbl_details.setContentsMargins(8, 6, 8, 6)
        self.lbl_details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details = QScrollArea(side)
        details.setWidgetResizable(True)
        details.setFrameShape(QFrame.Shape.StyledPanel)
        details.setWidget(self.lbl_details)
        sl.addWidget(details, 1)

        steps = QGridLayout()
        steps.setSpacing(6)
        self.btn_prev = QPushButton("◀  Previous", side)
        self.btn_next = QPushButton("Next  ▶", side)
        self.btn_open = QPushButton("Open Step", side)
        self.btn_done = QPushButton("✓  Mark Complete", side)
        steps.addWidget(self.btn_prev, 0, 0)
        steps.addWidget(self.btn_next, 0, 1)
        steps.addWidget(self.btn_open, 1, 0)
        steps.addWidget(self.btn_done, 1, 1)
        sl.addLayout(steps)
        self.split.addWidget(side)

        view_index = self.split.indexOf(self.view)
        for i in range(self.split.count()):
            self.split.setStretchFactor(i, 1 if i == view_index else 0)
        self.split.setCollapsible(view_index, False)
        self.split.setSizes([250, 850, 300] if editable else [1000, 300])

        self.btn_out.clicked.connect(lambda: self.view.zoom_by(1 / 1.15))
        self.btn_in.clicked.connect(lambda: self.view.zoom_by(1.15))
        self.btn_fit.clicked.connect(self.view.fit)
        self.view.zoomChanged.connect(lambda s: self.lbl_zoom.setText(f"{round(s * 100)}%"))
        self.btn_prev.clicked.connect(lambda _c=False: self.host.graph_prev())
        self.btn_next.clicked.connect(lambda _c=False: self.host.graph_next())
        self.btn_open.clicked.connect(lambda _c=False: self.host.graph_open_current())
        self.btn_done.clicked.connect(lambda _c=False: self.host.graph_complete_current())

        self._restore_layout()

    # ---- what the host calls ---------------------------------------------

    def rebuild(self):
        self.view.rebuild()
        name = (self.view._wf.name if self.view._wf is not None else "") or "Untitled Workflow"
        self.lbl_title.setText(name)
        self.setWindowTitle(f"Workflow Graph — {name}")

    def restyle(self):
        self.view.restyle()

    def request_fit(self):
        self.view.request_fit()

    def show_hint(self, text: str):
        self.view.show_hint(text)

    def set_details(self, text: str):
        if text != self.lbl_details.text():
            self.lbl_details.setText(text)

    def sync_palette(self, source):
        """Copy the dialog's tool tree (categories, icons, tips) into the palette."""
        if self.tool_tree is None or source is None:
            return
        self.tool_tree.clear()
        for i in range(source.topLevelItemCount()):
            item = source.topLevelItem(i).clone()
            self.tool_tree.addTopLevelItem(item)
            item.setFirstColumnSpanned(True)
            item.setExpanded(True)
        self._refilter()

    # ---- palette ----------------------------------------------------------

    def _refilter(self):
        if self.tool_tree is None:
            return
        needle = self.edt_search.text().strip().lower() if self.edt_search is not None else ""
        for i in range(self.tool_tree.topLevelItemCount()):
            parent = self.tool_tree.topLevelItem(i)
            shown = 0
            for j in range(parent.childCount()):
                child = parent.child(j)
                show = (not needle) or needle in child.text(0).lower() or needle in child.toolTip(0).lower()
                child.setHidden(not show)
                shown += int(show)
            parent.setHidden(shown == 0)

    def _on_tool_double_clicked(self, item, _column=0):
        cid = item.data(0, Qt.ItemDataRole.UserRole) if item is not None else None
        if cid:
            self.host.graph_add_tool(str(cid))

    # ---- placement --------------------------------------------------------

    def _settings(self):
        return getattr(self.host, "settings", None)

    def _restore_layout(self):
        s = self._settings()
        if s is None:
            return
        try:
            geo = s.value(f"{self._prefix}/graph_geometry")
            if geo:
                self._placed = bool(self.restoreGeometry(geo))
            state = s.value(f"{self._prefix}/graph_split")
            if state:
                self.split.restoreState(state)
        except Exception:
            pass

    def _save_layout(self):
        s = self._settings()
        if s is None:
            return
        try:
            s.setValue(f"{self._prefix}/graph_geometry", self.saveGeometry())
            s.setValue(f"{self._prefix}/graph_split", self.split.saveState())
        except Exception:
            pass

    def _place_first_time(self):
        """Nothing saved yet: as large as fits, centred over the dialog's screen."""
        self._placed = True
        owner = self.parentWidget()
        screen = (owner.screen() if owner is not None else None) or self.screen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        w = min(1400, int(avail.width() * 0.9))
        h = min(900, int(avail.height() * 0.9))
        self.resize(w, h)
        center = owner.frameGeometry().center() if owner is not None and owner.isVisible() else avail.center()
        x = min(max(avail.left(), center.x() - w // 2), avail.right() - w)
        y = min(max(avail.top(), center.y() - h // 2), avail.bottom() - h)
        self.move(x, y)

    # ---- widget events ----------------------------------------------------

    def showEvent(self, ev):
        if not self._placed:
            self._place_first_time()
        super().showEvent(ev)
        self.host._graph_flush()
        self.view.setFocus()

    def hideEvent(self, ev):
        if not ev.spontaneous():      # closed or hidden by us, not minimized
            self._save_layout()
        super().hideEvent(ev)

    def changeEvent(self, ev):
        super().changeEvent(ev)
        if ev.type() == QEvent.Type.WindowStateChange and not self.isMinimized() and self.isVisible():
            self.host._graph_flush()
