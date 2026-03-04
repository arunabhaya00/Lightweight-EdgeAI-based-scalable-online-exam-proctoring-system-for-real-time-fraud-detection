# modules/report.py — Report generation for all 4 components

import os
import time
from datetime import datetime

from config import (
    RESULT_DIR, MATCH_THRESHOLD, FACE_WINDOW_SECONDS, FACE_WINDOW_THRESHOLD,
    WINDOW_THRESHOLD_PERCENT, OVERALL_THRESHOLD_PERCENT,
    TEXT_MODEL_PATH, COCO_MODEL_PATH, FACE_MODEL_PATH,
    CONF_PHONE, CONF_TEXT, CONF_PERSON, TEMPORAL_THRESHOLD, MAX_MISSING_FRAMES,
    C3_CSV_LOG, C3_JSON_LOG,
    C4_ECAPA_MODEL_PATH, C4_YAMNET_CLASSIFIER, C4_SEGMENT_SECONDS,
    C4_SPEAKER_THRESHOLD, C4_INTEGRITY_PASS, C4_INTEGRITY_REVIEW
)


def generate_report(app, answer_vars, questions):
    student   = app.current_student or "Unknown"
    eye_mod   = app.eye
    face_mod  = app.face
    obj_mod   = app.obj
    audio_mod = app.audio
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if eye_mod:
        eye_mod.tracker.check_alerts()
        et        = eye_mod.tracker
        wp        = et.window_percentage()
        op        = et.overall_percentage()
        vt        = et.total_violation_time
        dur       = time.time() - et.start_time
        win_alert = et.window_alert_triggered
        ovr_alert = et.overall_alert_triggered
    else:
        wp = op = vt = dur = 0.0
        win_alert = ovr_alert = False

    eye_cheat   = win_alert or ovr_alert
    eye_verdict = "*** CHEAT ***" if eye_cheat else "NOT CHEAT"
    eye_reasons = []
    if win_alert:
        eye_reasons.append(
            f"5-min window violation {wp:.1f}% exceeded threshold {WINDOW_THRESHOLD_PERCENT}%")
    if ovr_alert:
        eye_reasons.append(
            f"Overall violation {op:.1f}% exceeded threshold {OVERALL_THRESHOLD_PERCENT}%")
    if not eye_reasons:
        eye_reasons.append(
            f"Both thresholds within limits  "
            f"(window {wp:.1f}% < {WINDOW_THRESHOLD_PERCENT}%,  "
            f"overall {op:.1f}% < {OVERALL_THRESHOLD_PERCENT}%)")

    if face_mod:
        total_w    = face_mod.total_windows
        cheat_w    = face_mod.cheat_windows
        c_ratio    = face_mod.cheat_ratio()
        face_cheat = face_mod.final_result()
        window_log = face_mod.window_log
    else:
        total_w = cheat_w = 0
        c_ratio    = 0.0
        face_cheat = False
        window_log = []

    face_verdict = "*** CHEAT ***" if face_cheat else "NOT CHEAT"
    face_reasons = []
    if face_cheat:
        face_reasons.append(
            f"Cheat ratio {c_ratio:.2f} exceeded threshold 0.30  "
            f"({cheat_w} of {total_w} windows flagged)")
    else:
        face_reasons.append(
            f"Cheat ratio {c_ratio:.2f} within threshold 0.30  "
            f"({cheat_w} of {total_w} windows flagged)")

    if obj_mod:
        s3       = obj_mod.get_summary()
        c3_cheat = s3["cheat"]
    else:
        s3       = {}
        c3_cheat = False

    c3_verdict = "*** CHEAT ***" if c3_cheat else "NOT CHEAT"
    c3_reasons = []
    if obj_mod and c3_cheat:
        if s3.get("phone_detected"):
            c3_reasons.append("Mobile phone detected during exam")
        if s3.get("notebook_detected"):
            c3_reasons.append("Suspicious text/notes detected during exam")
        if s3.get("unauth_detected"):
            c3_reasons.append("Unauthorized person(s) detected during exam")
    if not c3_reasons:
        c3_reasons.append("No prohibited objects or unauthorized persons detected")

    if audio_mod:
        s4             = audio_mod.get_summary()
        c4_score       = s4["integrity_score"]
        c4_verdict_str = s4["verdict"]
        c4_cheat       = s4["is_cheat"]
        c4_review      = (c4_verdict_str == "REVIEW")
        c4_total_segs  = s4["total_segments"]
        c4_suspicious  = s4["suspicious"]
        c4_wav         = s4["wav_path"]
        c4_report      = s4["report_path"]
    else:
        c4_score       = 100.0
        c4_verdict_str = "PENDING"
        c4_cheat       = False
        c4_review      = False
        c4_total_segs  = 0
        c4_suspicious  = 0
        c4_wav         = "N/A"
        c4_report      = "N/A"

    c4_verdict = "*** CHEAT ***" if c4_cheat else ("REVIEW" if c4_review else "NOT CHEAT")

    answered     = sum(1 for v in answer_vars.values() if v.get() not in ("", "__none__"))
    final_cheat  = eye_cheat or face_cheat or c3_cheat or c4_cheat
    final_status = "*** CHEATING SUSPECTED ***" if final_cheat else "NO SUSPICIOUS BEHAVIOR"
    final_reason = []
    if eye_cheat:  final_reason.append("Eye/Head component flagged")
    if face_cheat: final_reason.append("Face Verification component flagged")
    if c3_cheat:   final_reason.append("Object/Person Detection component flagged")
    if c4_cheat:   final_reason.append("Audio Integrity component flagged")
    if not final_reason:
        final_reason.append("All components within acceptable limits")

    W = 65

    lines = [
        "=" * W,
        "        EDGE AI EXAM SYSTEM  –  PROCTORING REPORT",
        "=" * W,
        f"  Student         : {student}",
        f"  Report Generated: {timestamp}",
        f"  Questions        : {answered} / {len(questions)} answered",
        "",
        "─" * W,
        "  [ COMPONENT 1 — EYE GAZE & HEAD POSE MONITORING ]",
        "─" * W,
        f"  {'Metric':<26}{'Value':<15}{'Threshold'}",
        f"  {'5-Min Window Violation':<26}{wp:.2f}%{'':<9}{WINDOW_THRESHOLD_PERCENT}%",
        f"  {'Overall Violation':<26}{op:.2f}%{'':<9}{OVERALL_THRESHOLD_PERCENT}%",
        f"  {'Total Violation Time':<26}{vt:.1f} s",
        f"  {'Exam Duration':<26}{dur:.1f} s",
        f"  {'Window Alert Triggered':<26}{'YES' if win_alert else 'NO'}",
        f"  {'Overall Alert Triggered':<26}{'YES' if ovr_alert else 'NO'}",
        "",
        f"  COMPONENT 1 VERDICT : {eye_verdict}",
        f"  Reason: {eye_reasons[0]}",
    ]
    for r in eye_reasons[1:]:
        lines.append(f"          {r}")

    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 2 — FACE VERIFICATION  (10-Second Window Voting) ]",
        "─" * W,
        f"  Window Size            : {FACE_WINDOW_SECONDS} seconds",
        f"  Window Violation Thresh: >{FACE_WINDOW_THRESHOLD*100:.0f}% of frames mismatched",
        f"  Final Cheat Threshold  : cheat_ratio > 0.30",
        f"  Match Threshold        : {MATCH_THRESHOLD} (strict)",
        f"  Total Windows Checked  : {total_w}",
        f"  Windows Flagged (CHEAT): {cheat_w}",
        f"  Cheat Ratio            : {c_ratio:.3f}",
        "",
    ]

    if window_log:
        lines.append("  Window-by-Window Log:")
        lines.append("  " + "-" * 44)
        lines.append("  {:>8}  {:>10}  {:>10}".format("Window", "Viol Ratio", "Verdict"))
        lines.append("  " + "-" * 44)
        for wnum, ratio, verdict in window_log:
            flag = "  <-- FLAGGED" if verdict == "CHEAT" else ""
            lines.append(f"  {wnum:>8}  {ratio:>10.3f}  {verdict:>10}{flag}")
        lines.append("  " + "-" * 44)
        lines.append("")

    lines += [
        f"  COMPONENT 2 VERDICT : {face_verdict}",
        f"  Reason: {face_reasons[0]}",
    ]

    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 3 — OBJECT & UNAUTHORIZED PERSON DETECTION ]",
        "─" * W,
    ]

    if obj_mod and s3:
        lines += [
            f"  COCO Model             : {COCO_MODEL_PATH}",
            f"  Text Model             : {TEXT_MODEL_PATH}",
            f"  Face Model             : {FACE_MODEL_PATH}",
            f"  Multi-Person Threshold : {TEMPORAL_THRESHOLD}s continuous",
            f"  Phone Conf Threshold   : {CONF_PHONE}",
            f"  Text Conf Threshold    : {CONF_TEXT}",
            f"  Face Conf Threshold    : {CONF_PERSON}",
            f"  Face Tracker Buffer    : {MAX_MISSING_FRAMES} frames",
            "",
            f"  Phone Detected         : {'YES' if s3.get('phone_detected') else 'NO'}",
            f"  Notes/Text Detected    : {'YES' if s3.get('notebook_detected') else 'NO'}",
            f"  Unauth Person Detected : {'YES' if s3.get('unauth_detected') else 'NO'}",
            f"  Objects Seen           : {', '.join(s3.get('detected_objects', [])) or 'none'}",
            f"  Total Violation Events : {s3.get('total_violations', 0)}",
            f"  Log File (CSV)         : {C3_CSV_LOG}",
            f"  Log File (JSON)        : {C3_JSON_LOG}",
        ]
        vlog = s3.get("violation_log", [])
        if vlog:
            lines += [
                "",
                "  Violation Event Log:",
                "  " + "-" * 50,
                "  {:>20}  {}".format("Timestamp", "Violation Type"),
                "  " + "-" * 50,
            ]
            for ev in vlog:
                lines.append(f"  {ev['time']:>20}  {ev['type']}")
            lines.append("  " + "-" * 50)
            lines.append("")
    else:
        lines.append("  Component 3 was not initialised or models not available.")
        lines.append("")

    lines += [
        f"  COMPONENT 3 VERDICT : {c3_verdict}",
        f"  Reason: {c3_reasons[0]}",
    ]
    for r in c3_reasons[1:]:
        lines.append(f"          {r}")

    lines += [
        "",
        "─" * W,
        "  [ COMPONENT 4 — AUDIO INTEGRITY & VOICE VERIFICATION ]",
        "─" * W,
        f"  ECAPA Model            : {C4_ECAPA_MODEL_PATH}",
        f"  YAMNet Classifier      : {C4_YAMNET_CLASSIFIER}",
        f"  Segment Length         : {C4_SEGMENT_SECONDS}s",
        f"  Speaker Threshold      : cosine dist < {C4_SPEAKER_THRESHOLD}",
        f"  Pass Threshold         : integrity >= {C4_INTEGRITY_PASS}%",
        f"  Review Threshold       : integrity {C4_INTEGRITY_REVIEW}–{C4_INTEGRITY_PASS-1}%",
        f"  Fail Threshold         : integrity < {C4_INTEGRITY_REVIEW}%",
        "",
        f"  Total Segments Analysed: {c4_total_segs}",
        f"  Suspicious Segments    : {c4_suspicious}",
        f"  Integrity Score        : {c4_score:.1f}%",
        f"  WAV Recording          : {c4_wav}",
        f"  Segment Report         : {c4_report}",
        "",
        f"  COMPONENT 4 VERDICT : {c4_verdict}",
        f"  Reason: integrity score {c4_score:.1f}%  "
        f"({'≥'+str(C4_INTEGRITY_PASS)+'% → clean' if not c4_cheat and not c4_review else str(C4_INTEGRITY_REVIEW)+'–'+str(C4_INTEGRITY_PASS-1)+'% → review' if c4_review else '<'+str(C4_INTEGRITY_REVIEW)+'% → cheat'})",
    ]

    lines += [
        "",
        "=" * W,
        "  COMBINED FINAL RESULT  (OR-gate: any component CHEAT → CHEAT)",
        "=" * W,
        f"  Component 1 – Eye & Head        : {eye_verdict}",
        f"  Component 2 – Face Verification : {face_verdict}",
        f"  Component 3 – Object / Person   : {c3_verdict}",
        f"  Component 4 – Audio Integrity   : {c4_verdict}",
        "  " + "─" * (W - 2),
        f"  FINAL VERDICT                   : {final_status}",
        f"  Reason                          : {' | '.join(final_reason)}",
        "=" * W,
        "",
    ]

    vid1 = getattr(app, "video_path",     None) or "N/A"
    vid2 = getattr(app, "video_path_obj", None) or "N/A"
    lines += [
        "",
        "─" * W,
        "  [ RECORDED VIDEOS & AUDIO ]",
        "─" * W,
        f"  Video 1 (Eye/Head + Face Verif) : {vid1}",
        f"  Video 2 (Object / Person Detect): {vid2}",
        f"  Audio   (Voice Integrity)       : {c4_wav}",
        "─" * W,
        "",
    ]

    report_str  = "\n".join(lines)
    report_path = os.path.join(
        RESULT_DIR, f"{student}_{int(time.time())}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    print(f"[Report] Saved to {report_path}")
    return report_str, report_path
