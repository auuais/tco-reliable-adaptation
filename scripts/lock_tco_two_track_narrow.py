"""Bind the narrow-versus-uncached signed engineering twins only after the registered development gate."""
from lock_tco_two_track_jobs import write_roster
from tco_two_track_protocol import MANIFEST, PUBLIC, ROOT, STUDY, digest, read


def main():
    gate=PUBLIC/"a_alternative_gate.json"
    if read(gate)["candidate"]!="Narrow":
        raise ValueError("Narrow was not selected by the registered alternative gate")
    base=MANIFEST.with_name(STUDY+"_engineering_v2.json")
    common=PUBLIC/"a_engineering.json"
    if not read(common)["passed"]:
        raise ValueError("Shared hybrid engineering has not qualified")
    twins=[dict(j) for j in read(base)["jobs"] if j["profile"]=="fixed"]
    if [j["arm"] for j in twins] != ["Refresh1","Refresh1Uncached"]:
        raise ValueError("The original full-width signed twins are incomplete")
    for j in twins:
        j["id"]=j["id"].replace("engineering_v2_","engineering_narrow_").replace("Refresh1","Narrow")
        j["arm"]=j["arm"].replace("Refresh1","Narrow")
        if j["steps"]!=2 or not j["qualification_saved_on_cpu"]:
            raise ValueError("Invalid qualification operating point")
    write_roster("engineering_narrow",twins,PUBLIC/"development_inputs.json",
           dict(selection_sha256=digest(gate),shared_qualification_sha256=digest(common),
                signed_twins_template_sha256=digest(base),
                qualification="Check all 208 first-step and 192 second-step signed factor gradients, 208 final factors, "
                "104 effective BA updates and 25 per-view readouts. Both cached boundaries must be exercised. "
                "Both Adam instances must show 16 early factor steps of one and 192 suffix factor steps of two."))


if __name__=="__main__":
    main()
