import matplotlib.pyplot as plt
import seaborn as sns

def barchart(x,y,title):
    """
    draws bar plot,
    x: labels
    y: feature importance or ranking
    """
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=x,y=y).set(title=title)
    return fig