import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

Window {
    id: win
    width: 200
    height: 60
    visible: true
    color: "transparent"

    // Wayland-friendly window flags (don’t override these from Python later)
    flags: Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint

    Rectangle {
        id: root
        anchors.fill: parent
        color: "#80000000"
        radius: 30
        border.color: "#555"
        border.width: 1

        // Bind directly to injected Python object (context property "backend")
        property double cpuUsage: backend ? backend.cpuUsage : 0.0
        property double ramUsage: backend ? backend.ramUsage : 0.0
        property double gpuUsage: backend ? backend.gpuUsage : 0.0
        property string appRamString: backend ? backend.appRamString : "0 MB"

        // Wayland-friendly drag handler
        MouseArea {
            anchors.fill: parent
            onPressed: win.startSystemMove()
            cursorShape: Qt.OpenHandCursor
        }

        component MiniGauge: Item {
            Layout.preferredWidth: 40
            Layout.preferredHeight: 40
            property color barColor: "#0f0"
            property double value: 0

            onValueChanged: gaugeCanvas.requestPaint()

            Canvas {
                id: gaugeCanvas
                anchors.fill: parent
                antialiasing: true
                onPaint: {
                    var ctx = getContext("2d");
                    var cx = width / 2;
                    var cy = height / 2;
                    var r = (width / 2) - 3;

                    ctx.reset();

                    ctx.beginPath();
                    ctx.arc(cx, cy, r, 0, 2*Math.PI);
                    ctx.lineWidth = 4;
                    ctx.strokeStyle = "#444";
                    ctx.stroke();

                    var start = -Math.PI/2;
                    var end = start + (value/100 * 2*Math.PI);

                    ctx.beginPath();
                    ctx.arc(cx, cy, r, start, end);
                    ctx.lineWidth = 4;
                    ctx.lineCap = "round";
                    ctx.strokeStyle = barColor;
                    ctx.stroke();
                }
            }

            Text {
                anchors.centerIn: parent
                text: Math.round(value) + "%"
                font.pixelSize: 10
                font.bold: true
                color: "#fff"
            }
        }

        RowLayout {
            anchors.centerIn: parent
            spacing: 15

            ColumnLayout {
                spacing: 2
                MiniGauge {
                    value: root.cpuUsage
                    barColor: root.cpuUsage > 80 ? "#ff4444"
                            : (root.cpuUsage > 50 ? "#ffbb33" : "#00C851")
                }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "CPU"
                    color: "#aaaaaa"
                    font.pixelSize: 9
                }
            }

            ColumnLayout {
                spacing: 2
                MiniGauge { value: root.ramUsage; barColor: "#33b5e5" }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "RAM"
                    color: "#aaaaaa"
                    font.pixelSize: 9
                }
            }

            ColumnLayout {
                spacing: 2
                MiniGauge { value: root.gpuUsage; barColor: "#aa66cc" }
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "GPU"
                    color: "#aaaaaa"
                    font.pixelSize: 9
                }
            }
        }
    }
}
