# The EP market: currency and bids

The agents of MAREX interact through a market of exploration packages (EPs): offers, bids, awards and completion reports (`epmarket.py`, `marex_policies.py`). This note describes how the bids are computed, since 2026-10-13.

## The currency

The unit of the currency is **one timestep of exploration budget**. Bids and payments are in this unit.

The offer carries a prize $R = 0.1 |A|$, proportional to the area of the EP. The prize is the offering agent's estimate of the value of the EP. It is sent in the offer, but it does not restrict the bids: there is no reserve price.

## The bid: the marginal cost

When a contractor receives an offer (`MAREX_Contractor.on_ep_offer`, `can_bid`), it computes two schedules with `ExplorationPackageSet.find_shortest_path_ep`:
- $S$, through its commitments;
- $S^+$, through its commitments and the offered EP.

Both schedules start where and when the EP that the contractor is currently exploring ends, or at its current position and the next timestep if it is not in an EP (`schedule_start`). The EP in progress is not part of either schedule.

- **Feasibility:** the contractor bids only if $S^+$ ends within its budget.
- **The bid:** $t_\text{end}(S^+) - t_\text{end}(S)$, the number of timesteps by which the EP lengthens the schedule (`schedule_end`).
  - For an idle contractor, this is the travel to the EP plus its lawnmower pattern. The lawnmower is the same for every contractor, so the nearest idle contractor bids lowest.
  - For a contractor with commitments, it is the detour that the EP causes.
- **Pending bids:** the EPs of the contractor's outstanding bids (bid on, not yet awarded) count as commitments. A contractor that receives several offers in the same round therefore cannot win more of them than its budget allows. This is pessimistic, because a bid that is later lost raised the other bids. But an auction lasts only one timestep.

## The award and the payment

The offering agent awards the EP to the lowest bid, breaking ties by the names of the bidders (`EPAgent.clear`), and pays the winning bid when the completion is reported (first-price). The payments are recorded in the ledgers, but no decision depends on them.

## Before 2026-10-13

Every contractor bid the prize itself. The tie-break by name then gave every EP to the first feasible contractor in name order: con-1 won 52 of the 54 EPs in the 20 replications of the clustered 200x200 setting. Also, the feasibility test started the path at the robot's current position, but at the time when its current EP ends. The new `schedule_start` fixes this.

The tests are in `src/test/test_marex_communication.py` (`TestMarginalCostBids`, `TestMarketProtocol`).
