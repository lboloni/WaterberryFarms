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

## Lost bids

The award goes only to the winner, so a contractor is never told that it lost an auction. The offering agent clears an offer `AWARD_DELAY_ROUNDS` = 2 communication rounds after it sends it (`epmarket.py`): the bids are sent in the round after the offer, and the award in the round after that. A bid that has not been awarded by then is lost. `EPAgent.expire_bids` moves it from `outstanding_bids` to `lost_bids`, so that it no longer counts as a commitment. The contractor calls it before every bid, and at every timestep.

Before 2026-10-14 (commit 123fa02), the lost bids were never removed. Every EP a contractor had ever lost then lengthened all its later schedules, and the share of the pioneer's offers without any bid rose from 10–18% to 34–35%.

## Contractors that create EPs (option `contractors-offer`)

The run-level key `marex-contractors-offer` (true/false, default false) is passed by `run_experiment.py` to every MAREX policy as `exp_policy["contractors-offer"]`. It is set in the 12 MAREX run files of `marex2027-run`, and not in their defaults, because a key in the defaults would change the configuration, and hence the reuse, of every run.

With the option on:
- **Streaks:** every robot tracks streaks of detections (`MAREX_Policy.track_streak`). A Contractor does so only for observations outside the EP it explores and its commitments, and a streak in progress is dropped when it enters one.
- **Offers:** every robot builds the EP around a finished streak in the same way (`build_offer`: bounding box, border of 3 cells, $r = 2$, prize $0.1|A|$). It does not offer it if it overlaps any EP it has heard offered or offered itself (`known_eps`).
- **The offering side** of the market (`before_send`, `on_ep_bid`, `on_ep_completed`) is in `MAREX_Policy`, and is the same for all robots.
- **Self-bids:** a Contractor bids on its own offer like the others, when it broadcasts it, but without a message (`on_own_offer_sent`). If it wins, the EP becomes a commitment without an award message (`on_self_award`), and its completion is settled in its own ledger without a message (`report_completion`).
- **The Pioneer** never bids, and never explores EPs.

With the option off, a Contractor never offers, and the Pioneer checks the overlap only against its own awarded deals, as before. The behavior is then that of the marginal-cost bids with the expiry of the lost bids.

The message sequence chart (`marex_graphics.choose_lifecycle`) only illustrates offers of the pioneer: the self-bid of a contractor is not a message, and could not be drawn.

## The award and the payment

The offering agent awards the EP to the lowest bid, breaking ties by the names of the bidders (`EPAgent.clear`), and pays the winning bid when the completion is reported (first-price). The payments are recorded in the ledgers, but no decision depends on them.

## Before 2026-10-13

Every contractor bid the prize itself. The tie-break by name then gave every EP to the first feasible contractor in name order: con-1 won 52 of the 54 EPs in the 20 replications of the clustered 200x200 setting. Also, the feasibility test started the path at the robot's current position, but at the time when its current EP ends. The new `schedule_start` fixes this.

The tests are in `src/test/test_marex_communication.py` (`TestMarginalCostBids`, `TestMarketProtocol`, `TestLostBids`, `TestContractorOffers`).

## The parameters of MAREX

The values that define MAREX are set by run-level keys `marex-<name>`, which `run_experiment.py` passes to every MAREX policy as `exp_policy["<name>"]`. Their defaults are in `MAREX_DEFAULTS` (`marex_policies.py`). Like `marex-contractors-offer`, they are written explicitly into the MAREX run files of `marex2027-run`, and not into their defaults, which would change the configuration, and hence the reuse, of the baselines too.

| key | default | meaning |
|---|---|---|
| `marex-ep-step` | 1 | the resolution $r$ of the EPs: the distance between the passes of their lawnmower. With a sensor that observes only the cell of the robot, $r = 1$ observes every row of an EP, $r = 2$ every second row. |
| `marex-ep-border` | 3 | the border, in cells, by which the bounding box of a streak is extended into an EP |
| `marex-prize-factor` | 0.1 | $\alpha$ in the prize $R = \alpha |A|$ |
| `marex-streak-below` | 1.0 | a streak is made of the consecutive observations whose TYLCV value is below this threshold (see below) |
| `marex-award-delay-rounds` | 2 | the communication rounds between the broadcast of an offer and its award (`EPAgent.award_delay_rounds`) |
| `marex-bid-evaluations` | 2000 | the bound of the EP path search when bidding, as a count of evaluated combinations |
| `marex-replan-evaluations` | 20000 | the same, when replanning |

### The streaks: `marex-streak-below`

The TYLCV map has three values: 1.0 for a healthy plant, 0.5 for an infected one, and 0.0 for a destroyed one (`environment.py`). A robot extends its streak of detections while it observes plants whose value is below `marex-streak-below` (`MAREX_Policy.track_streak`: `value < streak_below`). The first observation at or above the threshold ends the streak, and an EP is built around it (`build_offer`).

- **1.0 (the default):** the infected and the destroyed plants make up the streaks, as the VoI counts both as positives (`water_berry_farm.py`: `envfield.value < 1.0`).
- **0.5:** only the destroyed plants make up a streak; an infected plant ends it like a healthy one. Before the parameter was introduced (2026-10-14), the streaks were defined this way, by an equality with 0.0.

### The choice of the defaults

Until 2026-10-14 MAREX used $r = 2$ and the streaks of the destroyed plants only. A 2x2 ablation of the EP resolution and the streak threshold (the variants MAREX-r1, MAREX-inf and MAREX-r1-inf, replicated in `icc-2027-replicated`) led to the current defaults, $r = 1$ and the streaks of the infected and the destroyed plants (the former MAREX-r1-inf). The variants were then removed from the run files and from the paper; git keeps them.
