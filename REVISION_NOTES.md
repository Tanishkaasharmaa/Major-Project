# VOCAL code revision notes

## Scope

This revision makes targeted changes to the uploaded project. The existing
Pipecat processor layout, deterministic dialogue, SQLite mock database, and
E5/Chroma FAQ architecture are retained.

## Changes

### Diagnosis and dialogue
- Added `WifiState.MAC_NOT_ON_FILE` as an evidence-gap outcome. If the caller
  reports a new device but the record contains no registered MAC, the agent
  no longer claims that a mismatch has been established. It requests technical
  review instead.
- Updated the edge-case seed record `200010` to expect `mac_not_on_file`.
- Reworded `account_fine`: the response now reports that the account record
  shows creation and notification, while explicitly avoiding a claim that the
  caller's device is known to be registered or working.
- Device answers that conflict with their own wording now return unclear and
  trigger clarification (for example, "no, I bought a new phone" and
  "yes, the same phone").
- Added handling for negated device-change wording such as "no new phone".
- Added a configurable 45-second dialogue inactivity timeout. Noise/silence
  phrases do not extend the timeout. This is checked when the next user turn
  arrives; it is not a background timer.
- The provost/chairman approval mapping is unchanged because it remains an
  institutional assumption that requires Computer Centre confirmation.

### FAQ retrieval
- Retrieval checks every returned document's cosine distance independently.
  Only results at or below the configured threshold are sent to the LLM.
- New turns are detected by message-object identity rather than text equality,
  so identical consecutive questions are retrieved separately.
- Added `validate_faq_collection()` and startup validation for non-empty
  collection, cosine metric, embedding model, embedding dimension, and FAQ
  count metadata.
- `knowledge/ingest.py` now validates FAQ records and computes embeddings
  before replacing the existing collection. It is protected by a `main()`
  guard and writes the metadata required by startup validation.
- The agent and ingestion script both support `VOCAL_CHROMA_PATH`.

**Important:** the existing Chroma collection must be rebuilt once with the
revised ingestion script because the new startup check requires the
`embedding_model`, `embedding_dimension`, and `faq_count` metadata.

### Intent routing and experiments
- Mixed personal/infrastructure utterances are not rejected solely because
  they mention a router, access point, cable, or signal. The guard now
  distinguishes shared-infrastructure reports from personal account/device
  complaints.
- Added mixed examples to `intent_eval_set.py`.
- Runtime thresholds are read from `VOCAL_INTENT_MIN_SIM` and
  `VOCAL_INTENT_MARGIN`. The defaults are provisional (`0.80` and `0.02`);
  they are not presented as newly calibrated results.
- `test_intent_routing.py` prints PowerShell and bash commands to apply the
  calibrated values to the running agent.
- `test_intent_routing.py` and `test_knowledge_agent.py` now execute their
  expensive model-loading/calibration/benchmark work only when run directly.
  Importing them during pytest collection does not run those experiments.

## Verification performed in this environment

- All Python source files compile successfully.
- `python -m pytest -q`: **111 passed, 1 skipped**.
- `python phase2_demo.py --quiet`: **7/7 primary mock diagnosis states passed**.
- The skipped test module is `test_pipeline_smoke.py`: Pipecat is not installed
  in this review environment. The processor-level integration test must be
  run in the project's Colab environment with the project's Pipecat version.
- The retrieval processor's per-result filtering and repeated-identical-turn
  behaviour were also exercised with a small isolated test double.

## Not implemented in this revision

The project archive contains only the mock SQLite schema and no live student
database connection details, SMS/OTP provider contract or credentials, ticket
system API/schema, or authorised MAC-update endpoint. Therefore this revision
does not invent or simulate a production integration.

The following remain stubs and must stay disabled for real student data until
the relevant institutional interfaces and security requirements are supplied:
- OTP delivery and verification
- Real student database access
- Real support-ticket creation
- Real MAC-address update
- Human transfer / telephony integration

`requirements.txt` in the supplied archive was empty. It remains unchanged
rather than adding unverified package pins that could break the known Pipecat
environment. Record the exact working Colab package versions before making a
reproducible dependency lockfile.

## Commands to run in the project's Colab environment

From the project root:

1. Rebuild the FAQ collection:
   `python knowledge/ingest.py`

2. Run the unit tests:
   `python -m pytest -q`

3. Run the processor integration tests:
   `python -m pytest test_pipeline_smoke.py -v -s`

4. Calibrate intent routing (loads E5 and takes longer):
   `python test_intent_routing.py`

5. Apply the printed `VOCAL_INTENT_MIN_SIM` and `VOCAL_INTENT_MARGIN`
   environment values, then restart the agent.

6. Start the voice agent:
   `python agent.py`
