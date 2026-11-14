"""A6: verify upstream evidence, normalize holdout and render offline HTML.

This is a retrospective research summary, not a live production forecast.
B's earlier CV is an independent implementation audit, not comparable holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.phase2.result_adapter_a import matched_holdout, mean_pinball, normalize_predictions

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_training(folder: Path, a_data: Path) -> dict:
    meta_path = require_file(folder / "metadata.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("snapshot_cutoff") != "2026-11-06" or meta.get("training_cutoff") != "2024-12-17":
        raise ValueError("Unexpected A development/cutoff dates")
    if meta.get("selected_v2_on_cv") != "gbv2_12f_c1_a":
        raise ValueError("Unexpected CV-selected candidate; manual review required")
    if "not pristine" not in meta.get("caveat", "").lower():
        raise ValueError("Missing historical holdout caveat")
    for name in ("cv_predictions", "cv_metrics", "cv_ranking", "holdout_predictions", "holdout_metrics"):
        fp = require_file(folder / f"{name}.csv")
        item = meta["artifacts"][name]
        if sha256(fp) != item["sha256"] or len(pd.read_csv(fp)) != int(item["rows"]):
            raise ValueError(f"A3 artifact mismatch: {name}")
    manifest = json.loads(require_file(a_data / "data/phase2/latest_manifest.json").read_text(encoding="utf-8"))
    portfolio = require_file(a_data / "data/phase2/portfolio_returns.csv")
    if manifest.get("cutoff") != "2026-11-06" or sha256(portfolio) != manifest["portfolio_sha256"]:
        raise ValueError("A KBS snapshot mismatch")
    if meta["source_sha256"] != manifest["portfolio_sha256"]:
        raise ValueError("A training uses different source data")
    return meta


def verify_a4(folder: Path, train: Path, meta: dict) -> dict:
    record = json.loads(require_file(folder / "a4_manifest.json").read_text(encoding="utf-8"))
    if record.get("source_training_metadata_sha256") != sha256(train / "metadata.json"):
        raise ValueError("A4 does not belong to this A3 experiment")
    if record.get("source_data_sha256") != meta["source_sha256"]:
        raise ValueError("A4 is based on different snapshot")
    if record.get("selected_v2_on_cv") != meta["selected_v2_on_cv"]:
        raise ValueError("A4 changed the CV-selected model")
    if record.get("do_not_use_for_selection") is not True:
        raise ValueError("A4 holdout has an invalid selection label")
    raw_files = record.get("files", {})
    if not isinstance(raw_files, dict):
        raise ValueError("A4 manifest files must be a dictionary")
    files = {str(rel).replace("\\", "/"): digest for rel, digest in raw_files.items()}
    if len(files) != len(raw_files):
        raise ValueError("A4 manifest has duplicate normalized file paths")
    needed = {
        "holdout_backtests.csv", "cv_fold_metrics_verified.csv",
        "holdout_paired_comparison.csv", "holdout_exceptions.csv",
        "figures/actual_vs_quantile.png", "figures/cumulative_exceptions.png",
        "figures/cv_fold_pinball.png", "figures/holdout_pinball_comparison.png",
        "figures/rolling_60d_violation_rate.png",
    }
    if not needed.issubset(files):
        raise ValueError("A4 audit or chart missing")
    for rel, digest in files.items():
        if sha256(require_file(folder / rel)) != digest:
            raise ValueError("A4 checksum changed: " + rel)
    return record


def verify_b_audit(folder: Path, source_hash: str) -> tuple[dict, pd.DataFrame]:
    record = json.loads(require_file(folder / "audit_metadata.json").read_text(encoding="utf-8"))
    if record.get("a_source_sha256") != source_hash:
        raise ValueError("A5 crosscheck uses different A snapshot")
    if record.get("b_original_snapshot_bytes_available") is not False:
        raise ValueError("Unexpected B snapshot verification status")
    if record.get("b_snapshot_hash_verified_against_original_bytes") is not False:
        raise ValueError("B snapshot original hash status has changed")
    flags = record.get("protocol_flags", {})
    if flags.get("a_and_b_gb_forecasts_directly_comparable") is not False:
        raise ValueError("Incorrect claim of A/B direct comparability")
    path = require_file(folder / "b_baseline_audit.csv")
    if sha256(path) != record["audit_csv_sha256"]:
        raise ValueError("A5 B audit checksum changed")
    frame = pd.read_csv(path)
    if len(frame) != 3 or set(frame["status"]) != {"TECHNICAL_RECONSTRUCTION_PASS"}:
        raise ValueError("B baseline audit incomplete")
    return record, frame


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):,.{digits}f}"


def render_html(*, ranking: pd.DataFrame, hold: pd.DataFrame,
                back: pd.DataFrame, b: pd.DataFrame, candidate: str,
                percentage: float, picture_files: list[str]) -> str:
    def esc(s: object) -> str:
        return html.escape(str(s), quote=True)
    def table(frame: pd.DataFrame, fields: list[str], header_map: dict[str, str]) -> str:
        heads = "".join(f"<th>{esc(header_map.get(k,k))}</th>" for k in fields)
        rows = []
        for _, record in frame.iterrows():
            tds=[]
            for k in fields:
                v=record[k]
                if k in ("violation_rate",):
                    v=f"{100*float(v):.2f}%"
                elif k in ("cv_pinball", "pinball_loss", "mean_pinball_loss", "average_var", "p_uc", "p_ind", "p_cc"):
                    v=fmt(v)
                tds.append(f"<td>{esc(v)}</td>")
            rows.append("<tr>"+"".join(tds)+"</tr>")
        return "<table><thead><tr>"+heads+"</tr></thead><tbody>"+"".join(rows)+"</tbody></table>"
    sections = []
    captions = {
        "actual_vs_quantile.png": "Lợi suất thực tế, ngưỡng phân vị 5% và ngoại lệ",
        "cumulative_exceptions.png": "Số lần vi phạm VaR tích lũy",
        "cv_fold_pinball.png": "Pinball loss theo 5 folds (development)",
        "holdout_pinball_comparison.png": "So sánh pinball trên holdout lịch sử",
        "rolling_60d_violation_rate.png": "Vi phạm VaR theo cửa sổ 60 phiên",
    }
    for rel in picture_files:
        name = Path(rel).name
        # Images are copied under integration/figures and verified by manifest.
        sections.append(
            f'<figure><img src="figures/{esc(name)}" alt="{esc(captions[name])}" loading="lazy">'
            f'<figcaption>{esc(captions[name])}</figcaption></figure>'
        )
    return f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Portfolio VaR — A Phase 2 audit</title>
<style>
:root{{--ink:#17263e;--muted:#53647b;--line:#dde4ed;--accent:#1959ba;--paper:#f7f9fc}}
*{{box-sizing:border-box}} body{{font-family:system-ui,'Segoe UI',sans-serif;background:var(--paper);color:var(--ink);margin:0}}main{{max-width:1090px;margin:0 auto;padding:30px 16px 80px}}h1{{font-size:29px;margin:0 0 6px}}h2{{font-size:21px;margin-top:36px}}p{{line-height:1.55}}.lead{{color:var(--muted)}}.panel{{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;margin:12px 0;box-shadow:0 2px 10px #0f274008}}.stat{{display:flex;gap:10px;flex-wrap:wrap}}.stat .panel{{flex:1;min-width:185px}}.stat strong{{display:block;font-size:24px;color:var(--accent)}}.muted{{color:var(--muted)}}.warn{{border-left:4px solid #cc7620;background:#fffaf4}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line)}}th{{background:#eaf0fa}}.scroll{{overflow:auto}}figure{{margin:18px 0}}figure img{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}figcaption{{font-size:13px;text-align:center;color:var(--muted);margin-top:6px}}code{{font-family:ui-monospace,Consolas,monospace}}
</style></head><body><main><h1>Portfolio VaR — Phase 2</h1><p class="lead">A: kiểm tra độc lập · Dữ liệu KBS đến 06/11/2026 · Báo cáo kết quả lịch sử, không phải dự báo thời gian thực</p>
<div class="stat"><div class="panel"><span class="muted">Ứng viên chọn trên CV</span><strong>{esc(candidate)}</strong></div><div class="panel"><span class="muted">Cải thiện pinball holdout</span><strong>{percentage:.2f}%</strong></div><div class="panel"><span class="muted">Holdout được đối chiếu</span><strong>468 phiên</strong></div></div>
<div class="panel warn"><strong>Giới hạn sử dụng kết quả</strong><p>GBv2 C1 được chọn theo CV chỉ trên development trước 18/12/2024. Holdout từ 18/12/2024 đến 06/11/2026 được dùng để đánh giá hồi cứu, nhưng <b>đã trùng giai đoạn đánh giá từng công bố ở Phase 1</b>, không phải tập test hoàn toàn độc lập. G04 ở đây fit cố định theo fold/holdout; không đồng nhất với protocol Phase 1 refit mỗi ngày. Các dự báo B có giai đoạn và folds khác, không được so sánh hiệu năng trực tiếp với A.</p></div>
<h2>1. Lựa chọn bằng development CV</h2><div class="panel scroll">{table(ranking,['model','cv_pinball','n_features'],{'model':'Mô hình','cv_pinball':'Mean pinball CV','n_features':'Số features'})}</div>
<h2>2. Holdout cùng ngày (chỉ A 7F vs A 12F)</h2><div class="panel scroll">{table(hold,['model','n','violations','violation_rate','pinball_loss','average_var'],{'model':'Mô hình','n':'Số phiên','violations':'Vi phạm','violation_rate':'Tỷ lệ','pinball_loss':'Pinball','average_var':'Mean VaR'})}</div>
<h2>3. Kiểm định VaR</h2><div class="panel scroll">{table(back,['model','n','violations','violation_rate','p_uc','p_ind','p_cc'],{'model':'Mô hình','n':'Phiên','violations':'Vi phạm','violation_rate':'Tỷ lệ','p_uc':'Kupiec p','p_ind':'Christoffersen p','p_cc':'CC p'})}</div><p class="muted">p-value cao hơn 0,05 nghĩa là chưa bác bỏ giả thuyết tương ứng ở mức 5%, không chứng minh mô hình hoàn hảo.</p>
<h2>4. Kiểm chứng baseline của B (không so sánh holdout A/B)</h2><div class="panel scroll">{table(b,['model','prediction_rows','exception_count','violation_rate','mean_pinball_loss'],{'model':'Baseline B','prediction_rows':'Số dự báo','exception_count':'Vi phạm','violation_rate':'Tỷ lệ','mean_pinball_loss':'Pinball'})}</div>
<p class="muted">A tái lập HS/EWMA/GB của B từ dữ liệu KBS tương ứng; không có file bytes snapshot B ban đầu để xác nhận SHA256 gốc. B chưa tái lập GBv2 C1 trên cùng protocol của A.</p>
<h2>5. Biểu đồ</h2><div class="panel">{''.join(sections)}</div><p class="muted">Hình và bảng lấy từ dữ liệu A3/A4 đã kiểm SHA256. Xem integration_manifest.json để truy xuất nguồn và hash.</p></main></body></html>'''


def integrate(*, training: Path, backtesting: Path, b_audit: Path, data_root: Path,
              out: Path) -> dict:
    out = out.resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise FileExistsError("Integration output exists; never overwrite")
    a_meta = verify_training(training, data_root)
    a4_meta = verify_a4(backtesting, training, a_meta)
    b_meta, b_stats = verify_b_audit(b_audit, a_meta["source_sha256"])
    ranking = pd.read_csv(training / "cv_ranking.csv")
    holds = pd.read_csv(training / "holdout_metrics.csv")
    backs = pd.read_csv(backtesting / "holdout_backtests.csv")
    selected = a_meta["selected_v2_on_cv"]
    if len(ranking) != 3 or ranking.iloc[0]["model"] != a_meta["best_overall_on_cv"]:
        raise ValueError("CV ranking inconsistency")
    if selected not in set(ranking.model) or set(holds.model) != {"gb_G04_7f_a", selected}:
        raise ValueError("A holdout model selection mismatch")
    if set(backs.model) != set(holds.model):
        raise ValueError("A4 backtests for wrong model names")
    original = pd.read_csv(training / "holdout_predictions.csv", float_precision="round_trip")
    norm = normalize_predictions(
        original, source="A", protocol="fixed_fit_retrospective_holdout"
    )
    base, cand = matched_holdout(norm, "gb_G04_7f_a", selected)
    if len(base) != 468 or len(cand) != 468:
        raise ValueError("Unexpected A holdout length")
    if base.target_date.iloc[0] != pd.Timestamp("2024-12-18"):
        raise ValueError("Holdout start changed")
    if cand.target_date.iloc[-1] != pd.Timestamp("2026-11-06"):
        raise ValueError("Holdout end changed")
    baseline_loss = mean_pinball(base.actual_return, base.quantile_return)
    candidate_loss = mean_pinball(cand.actual_return, cand.quantile_return)
    comparison = pd.read_csv(backtesting / "holdout_paired_comparison.csv")
    if len(comparison) != 1 or comparison.iloc[0]["baseline"] != base.model.iloc[0] or comparison.iloc[0]["candidate"] != selected:
        raise ValueError("A4 paired comparison mismatch")
    percentage = 100 * (baseline_loss - candidate_loss) / baseline_loss
    if not np.isclose(percentage, comparison.iloc[0]["relative_loss_improvement_pct"], atol=1e-9):
        raise ValueError("A4 paired loss improvement does not reproduce")
    for name, frame in (("gb_G04_7f_a", base), (selected, cand)):
        metric = holds.loc[holds.model == name]
        back = backs.loc[backs.model == name]
        if len(metric) != 1 or len(back) != 1:
            raise ValueError("A holdout metrics missing")
        v = frame.violation.to_numpy(bool)
        for reported, expected in ((metric.iloc[0]["pinball_loss"], mean_pinball(frame.actual_return, frame.quantile_return)),
                                   (metric.iloc[0]["violation_rate"], v.mean()),
                                   (back.iloc[0]["mean_pinball_loss"], mean_pinball(frame.actual_return, frame.quantile_return)),
                                   (back.iloc[0]["violations"], v.sum())):
            if not np.isclose(float(reported), float(expected), atol=1e-12, rtol=0):
                raise ValueError("A3/A4 metric mismatch")
    if abs(candidate_loss - baseline_loss) < 1e-15:
        raise ValueError("Unexpected exact loss equality; manual review")
    figures = [
        "figures/actual_vs_quantile.png",
        "figures/cumulative_exceptions.png",
        "figures/cv_fold_pinball.png",
        "figures/holdout_pinball_comparison.png",
        "figures/rolling_60d_violation_rate.png",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="a6_tmp_", dir=out.parent) as tempdir:
        tmp = Path(tempdir)
        (tmp / "figures").mkdir()
        norm.to_csv(tmp / "normalized_holdout_predictions.csv", index=False, date_format="%Y-%m-%d", float_format="%.17g")
        summary = holds.merge(backs[["model", "p_uc", "p_ind", "p_cc"]], on="model", validate="one_to_one")
        summary["evaluation_scope"] = "A_retrospective_same_target_holdout_not_pristine"
        summary.to_csv(tmp / "model_comparison.csv", index=False, float_format="%.17g")
        for rel in figures:
            shutil.copyfile(backtesting / rel, tmp / rel)
        page = render_html(ranking=ranking, hold=holds, back=backs, b=b_stats,
                           candidate=selected, percentage=percentage, picture_files=figures)
        (tmp / "comparison.html").write_text(page, encoding="utf-8")
        artifacts = [p for p in tmp.rglob("*") if p.is_file()]
        output_hashes = {
            str(p.relative_to(tmp)).replace("\\", "/"): sha256(p)
            for p in sorted(artifacts)
        }
        manifest = {
            "owner": "A", "step": "A6_integrated_review_offline_dashboard",
            "as_of_snapshot": "2026-11-06",
            "training_cutoff": "2024-12-17",
            "holdout_first": "2024-12-18", "holdout_last": "2026-11-06",
            "selected_v2_on_development_cv": selected,
            "holdout_relative_pinball_improvement_pct": percentage,
            "holdout_same_dates": True,
            "holdout_selection_uses_only_cv": True,
            "holdout_not_pristine": True,
            "b_baselines_independently_reconstructed": True,
            "b_original_snapshot_hash_verified": False,
            "a_and_b_gb_predictions_directly_comparable": False,
            "b_has_independently_retrained_gbv2_c1": False,
            "live_issued_forecast": False,
            "limitations": [
                "A holdout overlaps previously reported Phase1 historical evaluation",
                "B snapshot byte hash not independently verified",
                "B G04 CV refits once per fold; A/B CV boundaries differ",
                "G04 A holdout uses fixed fit, not Phase1 daily-refit protocol",
                "Do not merge B metrics into A matched holdout ranking",
            ],
            "sources": {
                "a3_metadata_sha256": sha256(training / "metadata.json"),
                "a4_manifest_sha256": sha256(backtesting / "a4_manifest.json"),
                "a5_audit_metadata_sha256": sha256(b_audit / "audit_metadata.json"),
                "a5_b_baseline_audit_sha256": sha256(b_audit / "b_baseline_audit.csv"),
                "a_kbs_portfolio_sha256": a_meta["source_sha256"],
            },
            "output_files": output_hashes,
        }
        (tmp / "integration_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(str(tmp), str(out))
    return {"selected": selected, "a_holdout_rows": len(norm),
            "relative_pct": percentage, "b_audited_models": len(b_stats),
            "figures": len(figures), "html": out / "comparison.html", "output": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-dir", type=Path, required=True)
    ap.add_argument("--backtesting-dir", type=Path, required=True)
    ap.add_argument("--b-audit-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    result = integrate(training=a.training_dir.resolve(), backtesting=a.backtesting_dir.resolve(),
                       b_audit=a.b_audit_dir.resolve(), data_root=a.data_root.resolve(), out=a.out_dir.resolve())
    print("A HOLDOUT ROWS:", result["a_holdout_rows"])
    print("CV SELECTED:", result["selected"])
    print("HOLDOUT LOSS CHANGE (%):", f"{result['relative_pct']:.6f}")
    print("INDEPENDENTLY RECONSTRUCTED B BASELINES:", result["b_audited_models"])
    print("FIGURES:", result["figures"])
    print("HTML:", result["html"])
    print("A/B DIRECT COMPARABILITY: NO")
    print("A6 INTEGRATION PASS")


if __name__ == "__main__":
    main()
