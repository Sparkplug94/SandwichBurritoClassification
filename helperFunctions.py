def plotPCbar(pc,cols,ax): #function used for plotting principal component vectors as sorted bar charts
    pc_sort, cols_sort = zip(*sorted(zip(pc,cols))) #sort normalize dataframe column headers by magnitude of principal component
    ax.bar(cols_sort, pc_sort) #plot as bar chart

def scatterLabel(ax,x,y,labels, colors = 'blue'): #define helper function for plotting sandwiches and labeling them with text
    ax.scatter(x,y, color = colors)
    for i in range(len(x)):  # label them
        ax.text(x[i] - 0.1, y[i] + 0.03, '%s' % (str(labels[i])), size=8, zorder=1,
                 color='k')

def scatterLabel2(ax,x,y,labels, colors = 'blue'): #same as above, but with slightly different position of text
    ax.scatter(x,y, color = colors)
    for i in range(len(x)):  # label them
        ax.text(x[i] + 0.01, y[i], '%s' % (str(labels[i])), size=8, zorder=1,
                 color='k')

def label2color(labels): #converts cluster labels to colors for plotting. only works up until 7 clusters
    colors = []
    for label in labels:
        if label == 0:
            colors.append('red')
        elif label == 1:
            colors.append('green')
        elif label == 2:
            colors.append('blue')
        elif label ==3:
            colors.append('orange')
        elif label == 4:
            colors.append('purple')
        elif label == 5:
            colors.append('magenta')
        elif label == 6:
            colors.append('yellow')
        else:
            colors.append('black')
    return colors