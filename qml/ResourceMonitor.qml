// src/setiastro/qml/ResourceMonitor.qml
//
// System resource HUD — CPU / RAM / GPU gauges hosted inside
// SystemMonitorWidget (a QQuickWidget). Values come from the
// injected `backend` context property (setiastro.saspro.widgets.
// resource_monitor.ResourceBackend).
//
// Design notes:
//   * Gauges are drawn with QtQuick.Shapes (scene-graph, GPU-accelerated)
//     rather than the old Canvas (CPU-side rasterization).
//   * `Behavior on value` smooths the arc between polls so the widget
//     feels alive even though we sample at 2 Hz.
//   * HoverHandler + attached ToolTip give rich per-gauge detail on
//     hover WITHOUT capturing click events (so the Python side can still
//     see mouse presses for window dragging).

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Shapes 1.15

Rectangle {
    id: root
    width: 200
    height: 60
    color: "#80000000"
    radius: 30
    border.color: "#555"
    border.width: 1

    // ─── Bindings to the Python backend ─────────────────────────────────────
    property double cpuUsage:      backend ? backend.cpuUsage      : 0.0
    property double ramUsage:      backend ? backend.ramUsage      : 0.0
    property double gpuUsage:      backend ? backend.gpuUsage      : 0.0
    property string appRamString:  backend ? backend.appRamString  : "0 MB"
    property string ramString:     backend ? backend.ramString     : ""
    property string gpuName:       backend ? backend.gpuName       : "GPU"
    property string gpuMemString:  backend ? backend.gpuMemString  : ""

    // ─── Reusable circular gauge ────────────────────────────────────────────
    //
    // A ring gauge with:
    //   - a dim background ring
    //   - a colored value arc (0..100% -> 0..360° starting at 12 o'clock)
    //   - centered "%" label
    //   - hover tooltip with rich detail
    //
    // Value transitions animate with an ease-out curve so the arc glides
    // between samples instead of stepping.
    component MiniGauge: Item {
        id: gauge
        Layout.preferredWidth: 40
        Layout.preferredHeight: 40

        property color  barColor:    "#0f0"
        property double value:       0
        property string tooltipText: ""

        Behavior on value {
            NumberAnimation { duration: 400; easing.type: Easing.OutQuad }
        }

        Shape {
            id: shape
            anchors.fill: parent
            smooth: true
            antialiasing: true

            // Background ring
            ShapePath {
                strokeWidth: 4
                strokeColor: "#444"
                fillColor: "transparent"
                capStyle: ShapePath.RoundCap
                PathAngleArc {
                    centerX: shape.width / 2
                    centerY: shape.height / 2
                    radiusX: (Math.min(shape.width, shape.height) / 2) - 3
                    radiusY: (Math.min(shape.width, shape.height) / 2) - 3
                    startAngle: 0
                    sweepAngle: 360
                }
            }

            // Value arc (only drawn when there's something to show, so a
            // 0% gauge doesn't render a stray RoundCap dot at 12 o'clock)
            ShapePath {
                strokeWidth: 4
                strokeColor: gauge.barColor
                fillColor: "transparent"
                capStyle: ShapePath.RoundCap
                PathAngleArc {
                    centerX: shape.width / 2
                    centerY: shape.height / 2
                    radiusX: (Math.min(shape.width, shape.height) / 2) - 3
                    radiusY: (Math.min(shape.width, shape.height) / 2) - 3
                    startAngle: -90
                    // Clamp to a minimum visible sweep once value crosses ~0.5%
                    // (below that the arc is invisible and RoundCap would
                    // render a dot — hide it entirely instead).
                    sweepAngle: gauge.value > 0.5 ? gauge.value * 3.6 : 0
                }
            }
        }

        Text {
            anchors.centerIn: parent
            text: Math.round(gauge.value) + "%"
            font.pixelSize: 10
            font.bold: true
            color: "#fff"
        }

        // HoverHandler = hover detection that does NOT capture click events.
        // This is the key to letting drag work: the Python side sees the
        // mouse presses; the gauge just shows a tooltip on hover.
        HoverHandler {
            id: hoverHandler
        }

        ToolTip.visible: hoverHandler.hovered && gauge.tooltipText !== ""
        ToolTip.text:    gauge.tooltipText
        ToolTip.delay:   500
    }

    // ─── Layout ─────────────────────────────────────────────────────────────
    RowLayout {
        anchors.centerIn: parent
        spacing: 15

        ColumnLayout {
            spacing: 2
            MiniGauge {
                value: root.cpuUsage
                barColor: root.cpuUsage > 80 ? "#ff4444"
                        : (root.cpuUsage > 50 ? "#ffbb33" : "#00C851")
                tooltipText: "CPU: " + Math.round(root.cpuUsage) + "%"
            }
            Text {
                Layout.alignment: Qt.AlignHCenter
                text: "CPU"
                color: "#aaa"
                font.pixelSize: 9
            }
        }

        ColumnLayout {
            spacing: 2
            MiniGauge {
                value: root.ramUsage
                barColor: "#33b5e5"
                tooltipText: root.ramString !== ""
                    ? "RAM: " + root.ramString
                      + "  (this app: " + root.appRamString + ")"
                    : "RAM: " + Math.round(root.ramUsage) + "%"
            }
            Text {
                Layout.alignment: Qt.AlignHCenter
                text: "RAM"
                color: "#aaa"
                font.pixelSize: 9
            }
        }

        ColumnLayout {
            spacing: 2
            MiniGauge {
                value: root.gpuUsage
                barColor: "#aa66cc"
                tooltipText: {
                    var t = root.gpuName + ": " + Math.round(root.gpuUsage) + "%"
                    if (root.gpuMemString !== "")
                        t += "\nVRAM: " + root.gpuMemString
                    return t
                }
            }
            Text {
                Layout.alignment: Qt.AlignHCenter
                text: "GPU"
                color: "#aaa"
                font.pixelSize: 9
            }
        }
    }
}
