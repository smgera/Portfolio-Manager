import exchange_calendars as mcal
import pandas as pd

cal = mcal.get_calendar('NYSE')
print(f"First session: {cal.first_session}")
print(f"Last session: {cal.last_session}")
print(f"Calendar type: {type(cal)}")
print(f"Available methods: {[m for m in dir(cal) if not m.startswith('_') and 'session' in m.lower()]}")

# Try sessions property
try:
    sessions = cal.sessions
    print(f"\nSUCCESS accessing sessions property: {len(sessions)} total sessions")
    print(f"First few sessions: {sessions[:5].tolist()}")
    print(f"Last few sessions: {sessions[-5:].tolist()}")
    
    # Try to filter sessions
    start = pd.Timestamp('2023-01-01')
    end = pd.Timestamp('2023-01-31')
    filtered = sessions[(sessions >= start) & (sessions <= end)]
    print(f"Filtered sessions (2023-01): {len(filtered)} sessions")
except Exception as e:
    print(f"FAILED with sessions property")
    print(f"  Error: {type(e).__name__}: {e}")
