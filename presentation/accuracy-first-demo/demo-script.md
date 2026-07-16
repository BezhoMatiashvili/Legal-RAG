# Reliable demonstration script

This is the static fallback. It reads only persisted bundle results; no model, Qdrant, or
network connection is needed. Show the complete scorecard before or immediately after the
examples so no failure is hidden.

## Opening

This is an accuracy-first pre-production Georgian legal RAG. It answers only when evidence, identity, version, quotation, and completeness checks pass; otherwise it clarifies or abstains.

## Three predeclared exact-identity examples

### q169 — exact identity

Question: ცესკოს დადგენილება №9/2019 მერის რიგგარეშე არჩევნები კანდიდატის რეგისტრაცია

Outcome: `retrieval_identity_found`; result IDs: `['matsne:4513356']`.

Official URL: https://matsne.gov.ge/ka/document/view/4513356

Frozen snapshot quotation (offline audit, not legacy retrieval evidence):

> საქართველოს ცენტრალური საარჩევნო კომისიის

დადგენილება №9/2019

2019 წლის 20 მარტი

Frozen quote SHA-256: `10930c9abda970822b51d9cf57cb079f7509fa10d01f0bb10e10578d1778fb6e`; offsets `[37:119]`.

Canonical evidence ID/quotation/passage hash: `not_available_in_legacy_corpus`.
### q291 — exact identity

Question: განჩინება № 330310019002793689 შრომითი ბინადრობის ნებართვა

Outcome: `retrieval_identity_found`; result IDs: `['ecd:491610']`.

Official URL: https://ecd.court.ge/Decision#!?DecisionDocumentId=491610

Frozen snapshot quotation (offline audit, not legacy retrieval evidence):

> № 330310019002793689
         საქმე №3-ბ-1982-19

  გ ა ნ ჩ ი ნ ე ბ ა
  საქართველოს სახელით

23 იანვარი, 2020  წელი                                                                    ქ. თბილისი

      თბილისის სააპელაციო სასამართლოს
     ადმინისტრაციულ საქმეთა პალატა

Frozen quote SHA-256: `4beeaff9ce13fa58ecbe8217ef60a5080d4ed7e2f482c1cee654d3fb5665c4e8`; offsets `[0:267]`.

Canonical evidence ID/quotation/passage hash: `not_available_in_legacy_corpus`.
### q329 — exact identity

Question: კონსტიტუციური წარდგინება N787 მარიხუანის მოხმარება თავისუფლების აღკვეთა

Outcome: `retrieval_identity_found`; result IDs: `['constcourt:2173']`.

Official URL: https://constcourt.ge/ka/judicial-acts?legal=2173

Frozen snapshot quotation (offline audit, not legacy retrieval evidence):

> 273-ე მუხლის ის ნორმატიული შინაარსი, რომელიც ითვალისწინებს სისხლისსამართლებრივი სასჯელის სახით თავისუფლების აღკვეთის გამოყენების შესაძლებლობას ნარკოტიკული საშუალება „მარიხუანის“ მოხმარებისთვის, შესაძლოა მიჩნეულ იქნეს საქართველოს კონსტიტუციის მე-17 მუხლის მე-2 პუნქტის შეუსაბამოდ

Frozen quote SHA-256: `ddf9a6c4410383c1563d4863df647a03adce23c6c36183d4e6272294cd576fde`; offsets `[1013:1291]`.

Canonical evidence ID/quotation/passage hash: `not_available_in_legacy_corpus`.

## Ambiguity

Question `q121` returned `['matsne:106126', 'matsne:1282940']` and produced
`clarify` with reason `non_unique_document_number`. No answer text was emitted.

## Incomplete evidence

Question `q024` produced `abstain` with reason
`source_incomplete_or_unattested`. It was stopped by the frozen preflight completeness gate before
retrieval or answer composition; no source passage is displayed.

## Close with the complete scorecard

- Correct document identity: 15 / 15
- Deterministic repeats: 20 / 20
- Degraded/failed: run 1 `[]`, run 2 `[]`
- Question-set SHA-256: `61838610b10f4c319f22516405e0a1828b4bade019e97b356726d02e49fd64dd`

Legal correctness review: not yet independently adjudicated.
