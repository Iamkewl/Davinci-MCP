# DaVinci Resolve scripting API — sourced reference notes

This is the document the live backend was written from:
`packages/resolve-mcp/src/resolve_mcp/davinci_backend.py` and the offline harness in
`packages/resolve-mcp/tests/fake_resolve.py`. It lives in the repo so the backend's claims stay
checkable — every scripting method the backend calls appears below with the sources that
corroborate it, and `test_calls_only_documented_api` fails the build if the backend ever calls
something that isn't here.

Compiled 2026-09-12 from secondary/community mirrors of Blackmagic's official
"DaVinci Resolve Scripting README.txt" (v18-v21 era; the primary doc ships only
inside a Resolve/Studio install, so it cannot be downloaded standalone — the
sources below are widely-used, cross-consistent unofficial transcriptions).
No live Resolve was available to double-check against; treat anything marked
"single source" as lower confidence than anything corroborated by 2+ sources.

Sources fetched:
- https://raw.githubusercontent.com/leoweyr/DaVinci_Resolve_API_Docs/main/scripting_API/v18/scripting_API-v18.md (v18 mirror; MediaPool section missing/truncated on this mirror)
- https://gist.github.com/mhadifilms/2b84d469135315793220dbf2226cbe63 (v20.3 reference w/ code examples)
- https://gist.github.com/X-Raym/2f2bf453fc481b9cca624d7ca0e19de8 (v21.0.4)
- https://wiki.dvresolve.com/developer-docs/scripting-api (structured full mirror)
- https://github.com/CommandPost/ResolveCafe/blob/main/docs/developers/scripting.md
- https://resolvedevdoc.readthedocs.io/en/latest/readme_resolveapi.html
- https://www.muyanru.com/en/davinci/api/timelineitem
- https://deric.github.io/DaVinciResolve-API-Docs/ and https://extremraym.com/cloud/resolve-scripting-doc/ (general overview only, page summarized by fetch tool)

Note: all fetches went through a summarizing fetch tool, not a raw-text dump.
Where 3+ independent sources agree verbatim, confidence is high. Single-source
quotes are flagged.

## 1. Env bootstrap (per OS)

| Var | macOS | Windows | Linux |
|---|---|---|---|
| RESOLVE_SCRIPT_API | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting` | `%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting` | `/opt/resolve/Developer/Scripting` |
| RESOLVE_SCRIPT_LIB | `.../DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so` | `C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll` | `/opt/resolve/libs/Fusion/fusionscript.so` |
| PYTHONPATH | `$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/` | `%PYTHONPATH%;%RESOLVE_SCRIPT_API%\Modules\` | `$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/` |

Confirmed identically by 3 independent sources (deric/electron-rotoscope mirror,
extremraym mirror, mhadifilms gist). Key facts: Windows API path root is
**ProgramData**, not Program Files; the library is **fusionscript.dll directly
under the Resolve install root** (no `Developer\Scripting\Lib\` subpath, no
`.exe`); PYTHONPATH must point at the **`Modules` subfolder** of
RESOLVE_SCRIPT_API (that's where `DaVinciResolveScript.py` lives), not at
RESOLVE_SCRIPT_API itself.

When this was compiled, `.env.example` had all three wrong: a Program Files
root with an extra `API` suffix, a fabricated
`Lib\resolve_lite.exe` library path that appears in no mirror, and a `PYTHONPATH` missing the `\Modules\`
subfolder. All three were corrected against the table above, and
`davinci_backend._default_modules_dir()` derives the same path per OS.

## 2. Object/method inventory actually touched by davinci_backend.py

Resolve (from `DaVinciResolveScript.scriptapp("Resolve")`):
`GetProjectManager()`, `Quit()`. Real, no ambiguity. (`GetMediaStorage()`,
`OpenPage()`, `Fusion()` also documented but unused here.)

ProjectManager: `CreateProject(name)->Project|None`,
`LoadProject(name)->Project|None`, `SaveProject()->Bool`,
`GetCurrentProject()->Project`, `GetProjectListInCurrentFolder()->[str]`. All
real and match backend usage. (`DeleteProject`, `CloseProject`,
`CreateFolder` etc. exist but are unused/not needed here.)

Project: `GetName()`, `GetSetting(key)`, `SetSetting(key,val)`,
`GetMediaPool()`, `GetCurrentTimeline()`, `GetTimelineCount()`,
`GetTimelineByIndex(i)` (1-based), `SetCurrentTimeline(tl)->Bool`,
`SetRenderSettings({..})->Bool`, `AddRenderJob()->jobId`,
`StartRendering(jobId1,jobId2,...)` **or** `StartRendering([ids],
isInteractiveMode=False)` **or** `StartRendering(isInteractiveMode=False)`,
`GetRenderJobList()->[dict]`, `GetRenderJobStatus(jobId)->dict` (contains a
job-status field and a completion-percentage field; exact key casing
"JobStatus"/"CompletionPercentage" is only corroborated by the harness the
same agent wrote — **treat key names as single-source/unverified**, not
confirmed independently). `GetResolutionWidth/Height()` are used by the
backend but were not independently found in any mirror during this pass —
plausible (they exist on Project in most community references) but not
independently confirmed here.
**Missing from backend usage:** `SetCurrentRenderFormatAndCodec(format,
codec)->Bool` — real, documented, and never called anywhere in
`add_render_job`.

MediaPool: `GetRootFolder()->Folder`, `AddSubFolder(folder,name)->Folder|False`,
`SetCurrentFolder(folder)->Bool`, `ImportMedia([paths])->[MediaPoolItem]`,
`CreateEmptyTimeline(name)->Timeline`,
`AppendToTimeline([clipInfo,...])->[TimelineItem]` (clipInfo keys per 2
sources: `mediaPoolItem`, `startFrame`, `endFrame`, `mediaType` (1=video,
2=audio), `trackIndex`, `recordFrame` — matches backend's dict exactly),
`DeleteClips([MediaPoolItem,...])->Bool` (deletes items **from the media
pool/bin**), `DeleteTimelines([Timeline,...])->Bool`,
`MoveClips([clips],targetFolder)->Bool` (moves pool clips **between bins**,
not a timeline-repositioning API).

Folder: `GetName()`, `GetClipList()->[MediaPoolItem]`,
`GetSubFolderList()->[Folder]`. Real, matches.

MediaPoolItem (backend's "pool item"): `GetName()`, `GetClipProperty()->dict`
(returns keys like "File Path"; the exact key name is used correctly by
backend `prop.get("File Path", "")`). `GetUniqueId()->str` and
`GetMediaId()->str` both documented (single strong source, X-Raym gist) as
distinct methods — either would give a Resolve-stable id, unlike the backend's
current `hex(id(cv))` (Python object identity, dies on GC/re-fetch).

Timeline: `GetName()`, `GetSetting(key)`, `SetSetting(key,val)`,
`GetTrackCount(trackType:"video"|"audio"|"subtitle")->int`,
`GetItemListInTrack(trackType, index:1-based)->[TimelineItem]`,
`GetStartFrame()->int` ("frame number at the start of timeline" — this is the
absolute frame offset of the timeline's start timecode, e.g. 86400 for a
01:00:00:00 start at 24fps), `GetEndFrame()->int`, `AddMarker(frameId, color,
name, note, duration, customData)->Bool` (Timeline-level marker; frameId here
is confirmed by example code to be an absolute timeline offset),
**`DeleteClips([TimelineItem,...], rippleDelete:Bool)->Bool`** — confirmed by
3 independent sources (wiki.dvresolve.com structured page, mhadifilms gist
code example `timeline.DeleteClips([item1, item2], True)`, WebSearch summary)
as the real, documented way to remove a clip from a timeline.

TimelineItem: `GetName()`, `GetStart()->int` ("returns a position of first
frame" — absolute timeline frame, same coordinate space as
`Timeline.GetStartFrame()`), `GetEnd()->int`, `GetDuration()->int`,
`GetLeftOffset()/GetRightOffset()->int` (trim handles — noted in one forum
thread as unreliable on retimed clips), `GetSourceStartFrame()` /
`GetSourceEndFrame()->int` (confirmed real via targeted search, source in/out
of the underlying media), `GetMediaPoolItem()->MediaPoolItem`,
`SetProperty(key,val)->Bool`, `GetProperty(key)`, `AddMarker(frameId, color,
name, note, duration, customData)->Bool`, `GetUniqueId()->str` (single strong
source). **`Delete()` does NOT exist** — confirmed absent from the full
method enumeration in 4 independent sources (leoweyr v18 md — explicit
"not found"; the complete ordered method list pulled from the same mirror,
which lists every other Delete*-by-name method but no bare `Delete()`;
wiki.dvresolve.com structured page; muyanru.com TimelineItem page, which
explicitly lists item-scoped delete methods for markers/flags/versions/takes/
Fusion comps but nothing that deletes the item itself).

Documented `TimelineItem.SetProperty` key list (identical enumeration from 3
independent sources — resolvedevdoc/ResolveCafe/extremraym-derived mirrors):
`Pan, Tilt, ZoomX, ZoomY, ZoomGang, RotationAngle, AnchorPointX, AnchorPointY,
Pitch, Yaw, FlipX, FlipY, CropLeft, CropRight, CropTop, CropBottom,
CropSoftness, CropRetain, DynamicZoomEase, CompositeMode, Opacity, Distortion,
RetimeProcess, MotionEstimation, Scaling, ResizeFilter`. **No `Speed`, no
`FadeInStart/FadeInEnd/FadeOutStart/FadeOutEnd` key anywhere in any mirror
consulted.** `CompositeMode` and `RetimeProcess`/`Scaling`/etc. are documented
as "a value from the following constants" — i.e. symbolic constants on the
`resolve` object, not free-form strings. `resolve.COMPOSITE_*` constant names
enumerated by one source: `COMPOSITE_NORMAL, COMPOSITE_ADD, COMPOSITE_SUBTRACT,
COMPOSITE_DIFF, COMPOSITE_MULTIPLY, COMPOSITE_SCREEN, COMPOSITE_OVERLAY,
COMPOSITE_HARDLIGHT, COMPOSITE_SOFTLIGHT, COMPOSITE_DARKEN, COMPOSITE_LIGHTEN,
COMPOSITE_COLOR_DODGE, COMPOSITE_COLOR_BURN, ...` (single source for the exact
list, but the "constants not strings" framing is corroborated by the parallel
`RetimeProcess`/`Scaling` wording in the 3-source property table).

## 3. Hypothesis verdicts

`davinci_backend.py` cites these by number in its docstrings, so any live call can be traced
back to the evidence for it.

1. **`TimelineItem.Delete()` doesn't exist; real deletion is
   `Timeline.DeleteClips([items], ripple)`.** CONFIRMED (4 sources for
   non-existence, 3 for the real replacement).
2. **`AddMarker` needs a `duration` arg; signature is `(frameId, color, name,
   note, duration, customData)`.** CONFIRMED (3 sources, identical signature,
   `customData` explicitly called optional, `duration` not). frameId
   reference frame (clip-relative vs timeline-absolute) is NOT explicitly
   disambiguated in any mirror for the *TimelineItem* overload specifically
   (Timeline's own AddMarker is confirmed absolute); flagged as unverified —
   needs a live-Resolve smoke test.
3. **`SetProperty("CompositeMode", ...)` needs `resolve.COMPOSITE_*`
   constants, not title-case strings.** LIKELY CONFIRMED — docs describe the
   value as "a value from the following constants," consistent across the
   property-key table and the constants list. Backend passes strings like
   `"Add"`/`"Normal"` today.
4. **`Speed` / fade keys are not documented; retiming/fades may not be
   scriptable via SetProperty.** CONFIRMED absent from the documented key
   list in 3 independent sources.
5. **No documented API to reposition an existing timeline item in place.**
   CONFIRMED — no `SetStart`/`MoveClip`(for timeline items)/reposition method
   found anywhere; `MediaPool.MoveClips` only moves pool items between bins.
   Delete+re-append (via the *correct* `Timeline.DeleteClips`) is the only
   documented path, and it necessarily mints a new TimelineItem (loses
   grades/versions/Fusion comps attached to the old one).
6. **Ripple insert is not achievable via `AppendToTimeline`.** CONFIRMED —
   `AppendToTimeline`'s `recordFrame` places/overwrites at a position; no
   parameter or side documented that shifts subsequent items.
7. **`add_render_job`/render pipeline: format needs
   `SetCurrentRenderFormatAndCodec`.** CONFIRMED the method is real and
   documented; backend never calls it (code-level bug, not a docs question).
8. **`GetUniqueId()`/`GetMediaId()` exist and beat the (name,type,track,start)
   registry.** CONFIRMED on TimelineItem, Timeline, Project, and
   MediaPoolItem (`GetUniqueId` + `GetMediaId`).
9. **fps/resolution must be set via `Project.SetSetting` before first
   timeline, per-timeline via `Timeline.SetSetting`.** PARTIALLY CONFIRMED:
   example code in one source sets `timelineFrameRate` /
   `timelineResolutionWidth/Height` via `project.SetSetting(...)`; `Timeline`
   separately documents its own `GetSetting`/`SetSetting`. Ordering
   requirement ("must be before first timeline") not explicitly stated in any
   mirror consulted — treat as plausible but unverified without a live test.
10. **Env/bootstrap values.** CONFIRMED wrong in `.env.example` (see §1).
11. **`ProjectManager.SaveProject`, `GetProjectListInCurrentFolder`,
    `MediaPool.AddSubFolder`, `SetCurrentFolder`, `DeleteClips`,
    `DeleteTimelines`, `Resolve.Quit`; no Restart API.** All CONFIRMED real
    and matching backend usage/argument shapes. No `Restart`/`Relaunch`
    method found in any mirror — backend's `restart_app` aliasing to
    `quit_app` is an honest choice given that gap.
12. **`GetStart()`/timeline-start-offset frame math.** CONFIRMED:
    `TimelineItem.GetStart()`/`GetEnd()` return frame numbers in the same
    absolute coordinate space as `Timeline.GetStartFrame()` (i.e. include the
    timeline's start-timecode offset, commonly 86400 @24fps for a
    01:00:00:00 start). The backend never calls `Timeline.GetStartFrame()`
    and never subtracts it anywhere it reads or writes frame numbers —
    `_hydrate_timeline_state` converts `GetStart()` straight to
    `start_seconds` by dividing by fps, and `append_clip`/`move_clip` compute
    `recordFrame` as `round(seconds * fps)` with no offset added back. If a
    project's timeline does not start at frame 0 (Resolve's default is often
    01:00:00:00, i.e. frame offset present), every position the tool reports
    and writes will be off by that constant offset.
