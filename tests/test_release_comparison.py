import pytest

from pmmc.experiments.release_comparison import ARMS, compare_primary


def samples():
    def arm(a):
        evidence = [dict(query_id="q1", outcome="supported_pmmc" if a == "wr" else "refusal",
                         typed_score=1.0 if a == "wr" else 0.0, judge_score=None)]
        return dict(exploration_run_id="demo", exploration_variant=a, scope="main",
                    campaign_id="campaign-" + a, protocol_hash="protocol-" + a,
                    selection_hash="selected", phase="exploration", paper_result=False, groups=[dict(
                        dataset="memlens", answer_model="9b", selected_queries=1,
                        cells=[dict(evidence=evidence, evaluation_job_states={"completed": 1})],
                    )])
    return {a: arm(a) for a in ARMS}


def test_paired_readout_uses_same_queries_and_retains_unavailable_values():
    reports = samples()
    result = compare_primary(reports)["groups"][0]["effects"]["wr_minus_w"]
    assert result["typed_score"]["observed_mean_difference"] == 1
    assert result["judge_score"] == dict(observed_pairs=0, selected=1,
                                         observed_mean_difference=None)
    assert result["supported_pmmc_rate_difference"] == 1
    before = compare_primary(reports)
    reports["w"]["selection_ledger"] = {"settled_usd": "1.23"}
    assert compare_primary(reports) == before
    reports["r"]["groups"][0]["cells"][0]["evidence"][0]["query_id"] = "other"
    with pytest.raises(ValueError, match="denominator"):
        compare_primary(reports)


def test_partial_or_paper_report_cannot_be_a_comparison_arm():
    reports = samples()
    reports["w"]["groups"][0]["cells"][0]["evaluation_job_states"] = {"failed": 1}
    with pytest.raises(ValueError, match="not complete"):
        compare_primary(reports)
    reports["wr"]["paper_result"] = True
    with pytest.raises(ValueError, match="invalid"):
        compare_primary(reports)
