# Kaplan Violante 2010
## Final project for ECON 516

The code [KV 2010.jl](https://github.com/alpeters/KaplanViolante2010/blob/master/src/KV%202010.jl) replicates the heterogeneous agents life-cycle model from Kaplan & Violante (2010). While some of the details in the paper are sparse, I have tried to follow the same methods as much as possible. 
The only significant divergence from the paper is my use of standard budget constraints, whereas the authors apply mortality risk to the value of assets during retirement within the budget. More details on this are provided below.
Addtionally, the discretization method of the non-stationary income process is not fully described. I followed the Rouwenhorst method for non-stationary processes outlined in Fella, Gallipoli, & Pan (2019). I am adding the functions [rouwenhorst_ns](https://github.com/alpeters/QuantEcon.jl/blob/a270d70f420790db95b07ce8eae157a8fc83bd14/src/markov/markov_approx.jl#L166) and [simulate_ns](https://github.com/alpeters/QuantEcon.jl/blob/a270d70f420790db95b07ce8eae157a8fc83bd14/src/markov/mc_tools.jl#L466) to [markov_approx.jl](https://github.com/alpeters/QuantEcon.jl/blob/master/src/markov/markov_approx.jl) and [mc_tools.jl](https://github.com/alpeters/QuantEcon.jl/blob/master/src/markov/mc_tools.jl), respectively with the intention of contributing to the [QuantEcon.jl](https://github.com/QuantEcon/QuantEcon.jl) package. The code herein contains these functions so as to provide a standalone run-able version.
Some of the data used for calibration is not fully described in the article. I have documented below the data I used that hopefully closely resembles that used by the authors. Finally, I to-date, I have only implemented the zero borrowing constraint (ZBC), not the natural borrowing constraint (NBC).
Note that the code is still needs to be tidied. This is item 1 on the [todo list](https://github.com/alpeters/KaplanViolante2010/blob/master/todo.TODO).

## Results and Discussion
The following figures correspond to Figure 1 in the paper, with axes scaled similarly for ease of comparison:

![](/images/zbc_lifecycle_means.png)

![](/images/zbc_lifecycle_inequality.png)

It is immediately apparent that the level of savings is significantly lower than that in the paper. This leads to a sharp consumption drop at the beginning of retirement. One possible source of this discrepancy may be due to a different mortality risk, as the data was not described in the paper. It is also possible that there is a bug in my code that I haven't yet located. The alternative budget constraint specification that I used would cause an increase in savings, so cannot be to blame for this difference.


## Data
The data required was obtained and adapted as follows:
* PSID mortality info from from NCHS 1992 data (Chung 1994). In order to avoid a sharp drop in consumption at retirement due to a sudden morality risk, I scaled the data upwards to provide a smooth transition from the no-mortality risk during working periods to the retirements periods.

![](/images/survival_prob.png)

* Life-cycle earnings profile from PSID data up to 1992 (Huggett, Ventura, & Yaron 2006). In order to match the brief description provided in the paper, the data was squeezed from the original age range of 20-60 to 25-60 and the last two data points were scaled down to provide a smooth curve.

![](/images/kappa.png)


## Methods
* Rouwenhorst non-stationary discretization for permanent component of income process
* Rouwenhorst stationary discretization for transitory component of income process
* B-Spline interpolation using [Interoplations.jl](https://github.com/JuliaMath/Interpolations.jl)
* Endogenous grid method


## Derivations
I chose to employ a standard budget constraint for the retirement periods, rather than those described in the paper, which scale the asset by the mortality risk. This choice should induce a higher level of savings than that in the paper, as it removes the "risk" from the asset. The Euler equations used in my code, as well as my interpretation of those used in the paper, are derived in [this document](https://github.com/alpeters/KaplanViolante2010/blob/master/docs/KV2010.pdf).


## References
Chung, S. J. (1995). Formulas expressing life expectancy, survival probability and death rate in life table at various ages in US adults. International Journal of Bio-Medical Computing, 39(2), 209-217.

Fella, G., Gallipoli, G., & Pan, J. (2019). Markov-chain approximations for life-cycle models. Review of Economic Dynamics, 34, 183-201.

Huggett, M., Ventura, G., & Yaron, A. (2006). Human capital and earnings distribution dynamics. Journal of Monetary Economics, 53(2), 265-290.

Kaplan, G., & Violante, G. L. (2010). How much consumption insurance beyond self-insurance?. American Economic Journal: Macroeconomics, 2(4), 53-87.
