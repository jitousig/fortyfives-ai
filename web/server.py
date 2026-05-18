import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.game_session import GameSession, card_to_str
from web.room import Room, rooms

TRICK_DISPLAY_SECS = 4.0

app = FastAPI(title="Fortyfives")

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/")
async def index():
    # Never cache the HTML shell so the ?v= asset references can't be
    # pinned to an old build by the browser or service worker.
    return FileResponse(
        _static / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/sw.js")
async def service_worker():
    # Served from root so the worker's scope is the whole origin.
    return FileResponse(
        _static / "sw.js",
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


def _trick_complete_state(session: GameSession, seat: int = 0) -> dict:
    """`seat`'s filtered state with the last completed trick frozen in
    place for animation. Per-seat so it's safe to broadcast."""
    state = session.serialize_state_for(seat)
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
    # PR-A: a private solo room (one connection on seat 0; bots fill
    # 1-3) — single-player behaviour is unchanged, just expressed via
    # the Room model so PR-B can add real lobbies/seats. State sends go
    # through room.broadcast (per-seat filtered); with one seat-0
    # connection this is byte-identical to the old single send.
    room = Room.create_solo()
    room.add_connection(websocket, 0)
    session = room.session
    seat = 0  # the seat this connection controls

    # Advance past any opening AI turns (player 1 starts the auction)
    await _run_ai_turns(room, delay=0.15)
    await room.broadcast()

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "action":
                action = int(data["action"])
                pre_tricks = session.game.total_tricks_completed
                err = session.take_seat_action(seat, action)
                if err:
                    # Validation error → only the offending connection.
                    await websocket.send_json({"type": "error", **err})
                    continue

                if session.game.total_tricks_completed > pre_tricks:
                    # A trick just completed — freeze it on screen then advance
                    await room.broadcast(
                        lambda s: _trick_complete_state(session, s))
                    await asyncio.sleep(TRICK_DISPLAY_SECS)
                else:
                    # Normal card play or non-gameplay action
                    await room.broadcast()

                # The human's own card may have ended the hand — pause for
                # the summary popup before any next-hand AI turns.
                if session.hand_over:
                    await room.broadcast()
                    continue

                await _run_ai_turns(room, delay=0.5)
                await room.broadcast()

            elif msg_type == "continue_hand":
                session.continue_after_hand()
                await _run_ai_turns(room, delay=0.5)
                await room.broadcast()

            elif msg_type == "new_game":
                room.reset_session()
                session = room.session
                await _run_ai_turns(room, delay=0.15)
                await room.broadcast()

    except WebSocketDisconnect:
        pass
    finally:
        room.remove_connection(websocket)


# ---- Multiplayer rooms (PR-B1) -------------------------------------------
# Solo `/ws` above is untouched. These add lobby rooms: POST /room to
# create, then connect /ws/{code}. All game-mutating handlers run under
# room.lock so concurrent connections can't corrupt one shared game.

@app.post("/room")
async def create_room():
    return {"code": Room.create_room().code}


@app.websocket("/ws/{code}")
async def room_ws(websocket: WebSocket, code: str):
    await websocket.accept()
    room = rooms.get((code or "").upper())
    if room is None or room.solo:
        await websocket.send_json({"type": "error",
                                   "error": "Room not found"})
        await websocket.close()
        return

    room.add_connection(websocket, None)  # in lobby until a seat claimed
    await websocket.send_json(room.lobby_state_for(websocket))

    try:
        while True:
            data = await websocket.receive_json()
            t = data.get("type")

            if t == "join":
                room.join(websocket, str(data.get("name", "")))
                await room.broadcast_lobby()

            elif t == "claim_seat":
                err = room.claim_seat(websocket, data.get("seat"))
                if err:
                    await websocket.send_json({"type": "error", **err})
                await room.broadcast_lobby()

            elif t == "leave_seat":
                room.leave_seat(websocket)
                await room.broadcast_lobby()

            elif t == "start":
                async with room.lock:
                    err = room.start(websocket)
                    if err:
                        await websocket.send_json({"type": "error", **err})
                    else:
                        await room.advance_and_broadcast(delay=0.15)

            elif t == "action":
                async with room.lock:
                    seat = room.conns.get(websocket)
                    s = room.session
                    pre = s.game.total_tricks_completed
                    err = s.take_seat_action(
                        seat if isinstance(seat, int) else -1,
                        int(data["action"]))
                    if err:
                        await websocket.send_json({"type": "error", **err})
                    else:
                        if s.game.total_tricks_completed > pre:
                            await room.broadcast(room._frozen)
                            await asyncio.sleep(TRICK_DISPLAY_SECS)
                            await room.broadcast()
                        else:
                            await room.broadcast()
                        if s.hand_over:
                            await room.broadcast()
                        else:
                            await room.advance_and_broadcast(delay=0.5)

            elif t == "continue_hand":
                async with room.lock:
                    room.session.continue_after_hand()
                    await room.advance_and_broadcast(delay=0.5)

            elif t == "new_game":
                async with room.lock:
                    room.reset_session()
                    if room.started:
                        await room.advance_and_broadcast(delay=0.15)
                    else:
                        await room.broadcast_lobby()

    except WebSocketDisconnect:
        pass
    finally:
        room.remove_connection(websocket)
        # If a seated human dropped mid-game, their seat is now a bot —
        # nudge the game so it doesn't stall on the vacated seat.
        try:
            if room.code in rooms:
                if room.started:
                    async with room.lock:
                        await room.advance_and_broadcast(delay=0.3)
                else:
                    await room.broadcast_lobby()
        except Exception:
            pass


async def _run_ai_turns(room: Room, delay: float = 0.4):
    """Drive bot seats one turn at a time, broadcasting per-seat state
    updates between turns. A bot seat = any seat not in human_seats."""
    session = room.session
    safety = 0
    while (not session.game_over
           and session.game.current_player_id not in session.human_seats
           and safety < 120):
        pre_tricks = session.game.total_tricks_completed
        has_more = session.run_ai_turn()
        safety += 1

        if session.game.total_tricks_completed > pre_tricks:
            # A bot just completed a trick — freeze it on screen then advance
            await room.broadcast(lambda s: _trick_complete_state(session, s))
            await asyncio.sleep(TRICK_DISPLAY_SECS)
            await room.broadcast()
        else:
            await asyncio.sleep(delay)
            await room.broadcast()

        if not has_more:
            break
