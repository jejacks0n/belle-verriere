"""Sunrise/sunset, so the window sleeps at night and can shift its colors through the day."""
from __future__ import annotations

from datetime import datetime, timedelta

from .settings import Settings


def _sun_times(s: Settings, when: datetime):
    if s.latitude is None or s.longitude is None:
        return None
    try:
        from astral import Observer
        from astral.sun import sun
        times = sun(Observer(latitude=s.latitude, longitude=s.longitude), date=when.date(), tzinfo=when.tzinfo)
        return times["sunrise"], times["sunset"]
    except Exception as e:  # polar day/night, bad coordinates, missing package
        print(f"[daylight] {e}")
        return None


def now() -> datetime:
    return datetime.now().astimezone()


def is_awake(s: Settings, when: datetime | None = None) -> bool:
    if not s.daylight_only:
        return True
    when = when or now()
    st = _sun_times(s, when)
    if st is None:
        return 6 <= when.hour < 19  # no location set: rough default
    sunrise, sunset = st
    return (sunrise - timedelta(minutes=s.minutes_before_sunrise)) <= when <= \
           (sunset + timedelta(minutes=s.minutes_after_sunset))


def time_hint(s: Settings, when: datetime | None = None) -> str | None:
    """A short color nudge for the prompt based on where the sun is."""
    if not s.time_of_day_palette:
        return None
    when = when or now()
    st = _sun_times(s, when)
    if st:
        sunrise, sunset = st
        day = (sunset - sunrise).total_seconds() or 1
        f = (when - sunrise).total_seconds() / day
    else:
        f = (when.hour + when.minute / 60 - 7) / 12
    if f < 0.15:
        return "cool morning blues and pale gold"
    if f < 0.7:
        return None  # midday: leave the palette alone
    if f < 0.88:
        return "warm afternoon ambers"
    return "deep sunset reds, oranges and violet"


def status(s: Settings) -> dict:
    when = now()
    st = _sun_times(s, when)
    return {
        "awake": is_awake(s, when),
        "sunrise": st[0].strftime("%H:%M") if st else None,
        "sunset": st[1].strftime("%H:%M") if st else None,
        "time_hint": time_hint(s, when),
    }
