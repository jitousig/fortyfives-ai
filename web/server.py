import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.game_session import GameSession, card_to_str

TRICK_DISPLAY_SECS = 4.0

app = FastAPI(title="Fortyfives")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
async def index():
    return FileResponse(_static / "index.html")


def _trick_complete_state(session: GameSession) -> dict:
    """Return a state dict with the last completed trick frozen in place for animation."""
    state = session.serialize_state()
    game = session.game
    if game.last_completed_trick:
        state["current_trick"] = [
            card_to_str(c) if c is not None else None
            for c in game.last_completed_trick
        ]
    state["trick_animating"] = True
    state["last_trick_winner"] = game.last_completed_trick_winner
    return state


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
                pre_tricks = session.game.total_tricks_completed
                err = session.take_human_action(action)
                if err:
                    await websocket.send_json({"type": "error", **err})
                    continue

                if session.game.total_tricks_completed > pre_tricks:
                    # Human just completed a trick — freeze it on screen then advance
                    await websocket.send_json(_trick_complete_state(session))
                    await asyncio.sleep(TRICK_DISPLAY_SECS)
                else:
                    # Normal card play or non-gameplay action
                    await websocket.send_json(session.serialize_state())

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
        pre_tricks = session.game.total_tricks_completed
        has_more = session.run_ai_turn()
        safety += 1

        if session.game.total_tricks_completed > pre_tricks:
            # AI just completed a trick — freeze it on screen then advance
            await websocket.send_json(_trick_complete_state(session))
            await asyncio.sleep(TRICK_DISPLAY_SECS)
            await websocket.send_json(session.serialize_state())
        else:
            await asyncio.sleep(delay)
            await websocket.send_json(session.serialize_state())

        if not has_more:
            break
