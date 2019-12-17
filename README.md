# Kaplan Violante 2010
## Final project for ECON 516

This code replicates Kaplan & Violante (2010). While some of the details in the paper are sparse, I have tried to follow the same methods as much as possible. In particular, the discretization method of the non-stationary income process is unclear. I followed the Rouwenhorst method for non-stationary processes outlined in Fella, Gallipoli, & Pan (2019). I built this on top the [QuantEcon.jl](https://github.com/QuantEcon/QuantEcon.jl) package discretization functionality and intend to contribute it when ready.

## Results
[Insert Figure]

## Data
The data required was obtained as follows:
* PSID mortality info from from NCHS 1992 data (Chung 1994)
* Life-cycle earnings profile from PSID data up to 1992 (Huggett, Ventura, & Yaron 2006)

## Methods
* Rouwenhorst non-stationary discretization
*

## Derivations
First order conditions are derived in [this document](https://github.com/alpeters/KaplanViolante2010/blob/master/docs/KV2010.pdf)



## References
Chung, S. J. (1995). Formulas expressing life expectancy, survival probability and death rate in life table at various ages in US adults. International Journal of Bio-Medical Computing, 39(2), 209-217.

Fella, G., Gallipoli, G., & Pan, J. (2019). Markov-chain approximations for life-cycle models. Review of Economic Dynamics, 34, 183-201.

Huggett, M., Ventura, G., & Yaron, A. (2006). Human capital and earnings distribution dynamics. Journal of Monetary Economics, 53(2), 265-290.

Kaplan, G., & Violante, G. L. (2010). How much consumption insurance beyond self-insurance?. American Economic Journal: Macroeconomics, 2(4), 53-87.
