# MoMTSim Source-Code and Dataset Provenance Investigation

## 1. Objective

Investigate the upstream MoMTSim source code to understand how fraud labels are generated, which fraud scenarios are activated, and whether the source implementation can explain why MoMTSim V1 and V2 contain fraud labels exclusively on TRANSFER transactions.

## 2. Inspected source revision

Repository: https://github.com/aiinfinancegroup/MoMTSim

Local inspection directory: `/tmp/momtsim_upstream`

Inspected commit:

- Commit: `243fd12a22060f635a6383202a402b75f7c6ebc3`
- Date: 2024-07-25
- Message: `Delete screenshots/screenshot.txt`

These findings apply to this inspected revision. They must not automatically be generalized to other MoMTSim revisions.

## 3. Fraudster registration and configuration

`MoMTSimState.java`, method `initActors()`, registers and schedules:

- ThirdPartyFraudster
- FirstPartyFraudster

The configured number of fraudsters is divided approximately equally between these two groups.

The inspected `MoMTSim.properties` contains:

- `firstPartyFraudProbability=0.000`
- `thirdPartyFraudProbability=0.80`

`Parameters.java` loads these properties into the corresponding Java fields.

Consequently:

- Third-party fraud generation is active.
- First-party fraudsters are scheduled, but their fraud-generation probability is zero.
- DirectDepositFraudster, SplitDepositFraudster and RefundFraudster have implementations in the repository but are not registered in the inspected `initActors()` method.

## 4. Active third-party fraud scenario

The active scenario is implemented in `ThirdPartyFraudster.java`.

Its sequence is:

1. The scenario activates when a random value is below `thirdPartyFraudProbability`.
2. A target client and merchant are selected.
3. A PAYMENT transaction is generated and added to the transaction list.
4. If the payment succeeds, a TRANSFER from the victim to a mule is generated.
5. The transfer is explicitly labelled fraudulent with a probability of 0.30.
6. The payment and transfer transactions are submitted through `onTransactions()`.

The payment is not explicitly assigned `isFraud=true` in this method.

The transfer can also inherit the initiating client's existing fraud status through `Client.handleTransfer()`.

This implementation provides a plausible source-level explanation for the concentration of fraud labels on TRANSFER transactions.

## 5. First-party fraud

`FirstPartyFraudster.java` creates mule accounts and can generate transfers from faux accounts to a cash-out mule.

However, the inspected configuration sets `firstPartyFraudProbability=0.000`.

Therefore, the first-party fraud-generation path is not activated through its normal probability check under this configuration.

## 6. Other fraudster implementations

The repository also contains:

- DirectDepositFraudster
- SplitDepositFraudster
- RefundFraudster

Their implementations include fraud-labelled CASH_IN or TRANSFER transactions.

However, they are not registered in the inspected `MoMTSimState.initActors()` implementation. Their existence alone does not establish that these scenarios were active during dataset generation.

## 7. Mule cash-out handling

The inspected fraudster methods call `mule.fraudulentCashOut()`.

That method creates and returns a CASH_OUT transaction.

However, the inspected callers do not add the returned transaction to their transaction lists or independently submit it through `onTransactions()`.

Therefore, those particular mule cash-out transactions do not reach the normal raw-log collection path through these calls.

## 8. Transaction collection and raw export

The inspected transaction collection path is:

1. Actors submit transactions through `onTransactions()`.
2. `MoMTSimApp.onTransactions()` adds them to a shared transaction collection.
3. `MoMTSimApp.writeOutputStep()` retrieves the collection.
4. `Output.incrementalWriteRawLog()` writes each transaction.
5. `resetVariables()` replaces the collection after output.

No transaction-type or fraud-label filter was found in this raw-export path.

The step-aggregate exporter does filter failed transactions, but that is separate from raw transaction export.

## 9. Transaction serialization and schema discrepancy

`Transaction.java` tracks four separate status fields:

- `isFraud`
- `isFlaggedFraud`
- `isUnauthorizedOverdraft`
- `isSuccessful`

The `toString()` method serializes 13 values, including these four status fields.

However, `Output.java` declares a raw-log header with only 10 column names.

There is also a naming difference: the header uses `nameOrig` and `nameDest`, whereas the serialized transaction values use originator and destination IDs.

This is a header/row-width and naming inconsistency in the inspected source revision.

The upstream raw-log schema therefore cannot be assumed to match the published MoMTSim V1/V2 schemas without additional transformation or version evidence.

## 10. Comparison with the local datasets

Both locally investigated datasets contain these 10 columns:

- `step`
- `transactionType`
- `amount`
- `initiator`
- `oldBalInitiator`
- `newBalInitiator`
- `recipient`
- `oldBalRecipient`
- `newBalRecipient`
- `isFraud`

Observed fraud-label distribution:

- MoMTSim V1: all fraud labels occur on TRANSFER.
- MoMTSim V2: all fraud labels occur on TRANSFER.

The inspected upstream active third-party scenario creates PAYMENT and TRANSFER transactions, but only explicitly applies random fraud labelling to the TRANSFER.

This is consistent with the observed label concentration.

However, consistency is not proof of exact dataset provenance.

## 11. Provenance limitations

The investigation has not established:

1. That V1 and V2 were generated using commit `243fd12a`.
2. That the published datasets were exported directly from this repository without transformation.
3. That the exact inspected `MoMTSim.properties` configuration was used to generate both datasets.
4. Why the published dataset schema has 10 columns while the inspected serializer emits 13 values.
5. Whether the dataset generation process used additional scripts, post-processing, or a different source revision.

The V2 filename contains the timestamp `20240722202413`, while the inspected commit is dated 2024-07-25. This is a reason to investigate revision provenance further, not proof that the datasets were generated by an earlier revision.

## 12. Implications for fraud-detection modelling

The concentration of fraud labels on TRANSFER may reflect the simulator's active fraud-generation mechanism rather than a general property of real mobile-money fraud.

Therefore:

- Do not interpret transaction type as a universally reliable real-world fraud indicator.
- Avoid allowing transactionType to create a trivial classification shortcut.
- Evaluate TRANSFER-only discrimination separately.
- Treat strong balance and amount patterns as potential simulator artefacts.
- Preserve chronological evaluation and report temporal distribution drift.
- Do not claim production readiness based on synthetic-data performance.
- Seek independent datasets or external validation before making real-world fraud-detection claims.

## 13. Overall conclusion

The inspected MoMTSim source revision provides a plausible explanation for the TRANSFER-only fraud labels observed in V1 and V2.

Its active third-party fraudster scenario creates payment and transfer transactions, but explicitly assigns random fraud labels to the transfer. First-party fraud generation is disabled in the inspected configuration, while several other fraudster implementations are not registered.

The raw transaction exporter does not appear to filter by transaction type. Nevertheless, the serializer/header discrepancy and differences between upstream and published schemas mean that the exact provenance of the local V1 and V2 files remains unverified.

The findings should therefore be described as a source-code-supported explanation, not a proven reconstruction of the datasets' generation history.