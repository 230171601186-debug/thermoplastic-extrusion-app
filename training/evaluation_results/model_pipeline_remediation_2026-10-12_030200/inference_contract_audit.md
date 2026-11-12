# Inference contract audit

## Verified contract

- The Streamlit form has exactly eight customer-facing inputs: material type, grade, tube type, braid type, OD, wall thickness, ID, and required length.
- The feature mapping is asserted against `MODEL1_FEATURES`; candidate inference rejects missing or unexpected columns, missing/blank categories, missing/non-numeric/non-finite numeric inputs, non-positive geometry/length values, and geometry unless `|ID - (OD - 2 × wall)| < 0.25 mm`.
- `Braid_Type="None"` is an ordinary fitted category. An unseen nonblank category is accepted because the fitted encoder ignores unknowns, and a warning is returned rather than silently claiming a known category.
- The preprocessing objects live within each loaded pipeline. Model 1 and downstream input names/order and fitted categorical encoder policies are checked at bundle load. Both downstream estimators use only 17 pre-run/customer-predicted fields.
- Inference rejects malformed/non-finite predictions and Model 1 outputs outside existing *synthetic generator assertions*. Efficiency output is checked in `[0,100]` percentage points; scrap is checked in `[0,1]` fraction. These checks are not verified plant operating limits.
- The app converts scrap fraction to a percentage for display. Identical pipeline inputs are deterministic for these saved tree models; inference reload consistency is covered by the existing test suite.

## Gaps and presentation risks

1. **No numeric input support/domain check.** The inference contract enforces positivity and the ID/OD/wall identity, but has no upper/lower operating-support range for OD, wall, ID, or required length. As a reproducible diagnostic, valid-geometry dimensions OD=100 mm, wall=10 mm, ID=80 mm and length=1,000,000,000 m were accepted with no warning. Outputs happened to remain finite and within synthetic assertions. This proves no domain warning exists; it does not prove those dimensions or outputs are operationally valid. No Polyhose-specific range should be added without approved data/specifications.
2. **“Recommended Process Parameters” wording conflicts with the synthetic proof-of-concept notice.** The app also says estimates are not validated factory settings or production recommendations, but the section title can still imply actionable settings. This is a presentation/governance risk, not a numerical inference defect. Keep all current text unchanged in this phase.
3. **Candidate-versus-production status is not obvious in the UI.** The app explicitly loads candidate artifacts in code, while `models/model_metadata.json` describes different production-directory models. The technical choice is unambiguous; approval intent is not. The eight-field UI and estimate disclaimer remain explicit.
4. **Unseen categories are handled safely in software but do not imply semantic validity.** `OneHotEncoder(ignore)` prevents a crash and warning is emitted; it cannot make a previously unseen material/grade/tube combination trustworthy.

## No inference contract defect reproduced

The eight-to-nine-to-two flow, feature order, units, non-finite guards, documented synthetic output bounds, literal `None` braid, and error cases are covered by 16 existing inference tests plus four isolated pipeline tests. Existing app helper tests also pass. No training/inference transformation mismatch or post-production feature use was found in the active inference code.

## Safe follow-up

Do not invent bounds from generic extrusion knowledge. With process owners, define the supported product domain using approved specifications and representative historical records; then add boundary validation/warnings and tests. Before demo use, label the active bundle as candidate/demo-only and adjust presentation wording only with explicit authorization.
