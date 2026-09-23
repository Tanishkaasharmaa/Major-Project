from livekit import api
import os, sys

LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
ROOM_NAME = "phase0-test-room"

def make_token(identity: str) -> str:
    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME))
    )
    return token.to_jwt()

if __name__ == "__main__":
    identity = sys.argv[1] if len(sys.argv) > 1 else "test-user"
    print(make_token(identity))