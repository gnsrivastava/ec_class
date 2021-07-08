# Aim:
**To develop a machine learning model for the prediction of mixed EC class substrates**

# Background
> In biology, enzymes are categorized based on their action mechanism. For example EC1 which is class 1 type of enzyme represents oxidoreductase enzyme family. The class itself is then divided into multiple categories based on the specific action of a protein on substrate active group(s).
> Most of the enzyme sunstrate reactions fall under either EC1 or EC2 class of enzyme reactions and number of reactions and corresponding substrates go down drastically from EC3 - EC7. 
> In the EC class substrate data there is a huge class imbalance and we require a technique to handle the imbalance.
> The EC substrate data can be devided into pure EC class molecules and mixed EC class molecules. Pure EC class molecules are the reactants that only belong to one of the EC(x) and are exclusive to that EC class.
On the other hand, the mixed class molecules can belong to more than one EC class at the same time.

# Method
## Data preprocessing
**To Do list**
1. Find minority class(s)
2. Compute majority to minority label weight (ratio)
3. Define a function for upsampling of minority class based on K-neighbors
4. Augment the orifinal data using synthetic data to counter label imbalance  
