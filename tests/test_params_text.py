import pytest

from karak.stages.params_text import (
    parse_csv, parse_feature, parse_int_list, parse_names, parse_rule,
)


def test_parse_csv_strips_and_drops_empty():
    assert parse_csv(" Ca, Mg ,,BSE ") == ["Ca", "Mg", "BSE"]
    assert parse_csv("") == [] and parse_csv(None) == []


def test_parse_int_list():
    assert parse_int_list("2, 7", "target_phases") == [2, 7]
    with pytest.raises(ValueError, match="target_phases"):
        parse_int_list("2, x", "target_phases")


def test_parse_names():
    assert parse_names("0: Ilmenite (FeTiO₃); 1: Silica; 8: Fe Oxyhydroxide", "names") == {
        0: "Ilmenite (FeTiO₃)", 1: "Silica", 8: "Fe Oxyhydroxide"}
    with pytest.raises(ValueError, match="names"):
        parse_names("0 Ilmenite", "names")
    with pytest.raises(ValueError, match="names.*twice"):
        parse_names("0: A; 0: B", "names")


def test_parse_rule():
    assert parse_rule("Fe-K > 0.6 & Ca < 0.10", "rule") == [
        ("Fe-K", ">", 0.6), ("Ca", "<", 0.10)]
    assert parse_rule("Si >= 1", "rule") == [("Si", ">=", 1.0)]
    for bad in ("Fe-K 0.6", "Fe-K == 0.6", "Fe-K > abc", ""):
        with pytest.raises(ValueError, match="rule"):
            parse_rule(bad, "rule")


def test_parse_feature():
    assert parse_feature("Ca") == ("channel", ("Ca",))
    assert parse_feature("BSE") == ("bse", ())
    assert parse_feature("Ca/(Ca+Mg)") == ("ratio", ("Ca", "Mg"))
    with pytest.raises(ValueError, match="ratio"):
        parse_feature("Ca/(Mg+Fe)")      # numerator must be the first term
