import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.game_session import GameSession

app = FastAPI(title="Fortyfives")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


@app.websocket("/ws")
async def game_ws(websocket: WebSocket):
    await websocket.accept()
    session = GameSession(human_player=0)

    # Advance past any opening AI turns (player 1 starts the auction)
    await _run_ai_turns(websocket, session, delay=0.15)
    await websocket.send_json(session.serialize_state())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "action":
                action = int(data["action"])
                err = session.take_human_action(action)
                if err:
                    await websocket.send_json({"type": "error", **err})
                    continue
                # Send state immediately so the human's card disappears from hand
                await websocket.send_json(session.serialize_state())
                # Then run AI turns with visible delays
                await _run_ai_turns(websocket, session, delay=0.5)
                await websocket.send_json(session.serialize_state())

            elif msg_type == "new_game":
                session = GameSession(human_player=0)
                await _run_ai_turns(websocket, session, delay=0.15)
                await websocket.send_json(session.serialize_state())

    except WebSocketDisconnect:
        pass


async def _run_ai_turns(websocket: WebSocket, session: GameSession, delay: float = 0.4):
    """Drive AI players one turn at a time, sending state updates in between."""
    safety = 0
    while (not session.game_over
           and session.game.current_player_id != session.human_player
           and safety < 120):
        has_more = session.run_ai_turn()
        safety += 1
        await asyncio.sleep(delay)
        await websocket.send_json(session.serialize_state())
        if not has_more:
            break
