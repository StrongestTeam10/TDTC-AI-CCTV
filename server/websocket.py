"""
server/websocket.py - WebSocket 클라이언트 커넥션 관리자 및 메시지 브로드캐스트 유틸리티
"""

from typing import List, Dict, Any
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WebSocket Connected] 현재 연결 수: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WebSocket Disconnected] 현재 연결 수: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        dead_connections = []
        for connection in list(self.active_connections):
            try:
                if connection.client_state.name == "CONNECTED":
                    await connection.send_json(message)
                else:
                    dead_connections.append(connection)
            except Exception:
                dead_connections.append(connection)

        for dead_conn in dead_connections:
            self.disconnect(dead_conn)


manager = ConnectionManager()
