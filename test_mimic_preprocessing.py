from pathlib import Path

from scripts.prepare_mimic_ecg_ppg import choose_channels, infer_subject_id
from scripts.download_mimic_paired_pilot import parse_master, parse_physical


def test_choose_channels_prefers_lead_ii_and_pleth():
    ecg, ppg = choose_channels(["V", "PLETH", "II", "ABP"])
    assert ecg == 2
    assert ppg == 1


def test_choose_channels_rejects_missing_pair():
    assert choose_channels(["II", "ABP"])[1] is None
    assert choose_channels(["PLETH", "ABP"])[0] is None


def test_infer_subject_id_from_mimic_directory():
    path = Path("mimic3wdb-matched/1.0/p00/p000020/3544749_0001.hea")
    assert infer_subject_id(path) == "000020"


def test_parse_wfdb_master_and_paired_physical_header():
    master = "record/3 3 125 9000\nlayout 0\n~ 10\nsegment_0001 8990\n"
    assert parse_master(master) == ["layout", "segment_0001"]
    physical = (
        "segment_0001 3 125 7500\n"
        "segment_0001.dat 80 200/mV 8 0 0 0 0 II\n"
        "segment_0001.dat 80 10/NU 8 0 0 0 0 PLETH\n"
        "segment_0001.dat 80 1/mmHg 8 0 0 0 0 ABP\n"
    )
    parsed = parse_physical(physical)
    assert parsed["fs"] == 125.0
    assert parsed["signal_length"] == 7500
    assert parsed["dat_files"] == ["segment_0001.dat"]
