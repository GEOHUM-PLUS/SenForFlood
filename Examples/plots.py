import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# Source: https://stackoverflow.com/a/68510722
def smooth(scalars: list[float], weight: float) -> list[float]:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def plot_curves(data:dict, steps:list[int], smoothed:bool=False, save_path:str=None, show:bool=True, title:str='Training Curves') -> None:
    f, ax = plt.subplots(int(len(data)/2)+len(data)%2,2, figsize=(12, 4*int(len(data)/2)+len(data)%2))

    for k in range(len(data.keys())):
        ax.flatten()[k].title.set_text(list(data.keys())[k])
        curve_counter=0
        for curve in data[list(data.keys())[k]].keys():
            if smoothed:
                ax.flatten()[k].plot(steps, data[list(data.keys())[k]][curve], color=TABLEAU_COLORS[list(TABLEAU_COLORS.keys())[curve_counter]], alpha=0.3, linestyle='--')
                ax.flatten()[k].plot(steps, smooth(data[list(data.keys())[k]][curve], 0.9), label=curve, color=TABLEAU_COLORS[list(TABLEAU_COLORS.keys())[curve_counter]])
            else:
                ax.flatten()[k].plot(steps, data[list(data.keys())[k]][curve], label=curve, color=TABLEAU_COLORS[list(TABLEAU_COLORS.keys())[curve_counter]])
            curve_counter+=1
        ax.flatten()[k].legend()
        ax.flatten()[k].grid()
    
    plt.suptitle(title)
    
    plt.tight_layout()

    if not save_path is None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()