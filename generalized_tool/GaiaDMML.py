"""
Data Mining the Gaia Database and using Machine Learning
Author: Laura Vannozzi
"""

import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import axes3d, Axes3D


"""
Data Mining including displaying plots of the data
and trimming the data to compare against the clustering algorithm.
"""

def distance(x):
    if x <= 0:
        return 0
    else:
        return 1/x

def x(R, b, l):
    return R * math.cos(b*(math.pi/180)) * math.cos(l*(math.pi/180))

def y(R, b, l):
    return R * math.cos(b*(math.pi/180)) * math.sin(l*(math.pi/180))

def z(R, b):
    return R * math.sin(b*(math.pi/180))

def absmag(m, d):
    return (((math.log10(d) * 5)*-1)+5)+m

def badToGood(bad, everything):
    return bad/everything

#https://pages.uoregon.edu/soper/Light/luminosity.html
def luminosity(m, d):
    return 4*math.pi*math.pow(d, 2)*m

def solar_lum(l):
    return l/(3.828*math.pow(10, 26))

#σ = 5.670374419...×10−8 W⋅m−2⋅K−4
def radius(l, t):
    const = 5.67 * math.pow(10, -8)
    ans = l/(4*math.pi*const*math.pow(t, 4))
    return math.log10(math.sqrt(ans))


def create_df(csvfile):
    """

    :param csvfile: Needs to contain parallax in mas, g-r and b-r color indexes, g mean magnitude of each star
    :return:
    """
    df = pd.read_csv(csvfile)
    if 'parallax' in df.columns:
        df = df.dropna(subset=['parallax'])
        df.loc[:, 'parallax_arcsec'] = df['parallax'].apply(lambda x: x * .001)

        if 'b' in df.columns and 'l' in df.columns:
            df.loc[:, 'R'] = df['parallax_arcsec'].apply(distance)
            df.loc[:, 'x'] = df.apply(lambda r: x(r['R'], r['b'], r['l']), axis=1)
            df.loc[:, 'y'] = df.apply(lambda r: y(r['R'], r['b'], r['l']), axis=1)
            df.loc[:, 'z'] = df.apply(lambda r: z(r['R'], r['b']), axis=1)

        else:
            print("Need columns \'b\' and \'l\' to calculate x, y, and z coordinates.")
            exit()

        return df
    else:
        print("Need column \'parallax\' to calculate x, y, and z coordinates.")
        exit()
    #df.loc[:, 'magnitude'] = df.apply(lambda r: absmag(r['phot_g_mean_mag'], r['R']), axis=1)
    #df.loc[:, 'luminosity'] = df.apply(lambda r: luminosity(r['phot_g_mean_mag'], r['R']), axis=1)
    #df.loc[:, 'solar_luminosity'] = df.apply(lambda r: solar_lum(r['luminosity']), axis=1)
    #df.loc[:, 'radius'] = df.apply(lambda r: radius(r['luminosity'], r['teff_val']), axis=1)


def distance_plot(df, csvfile):
    fig = plt.figure()
    axp = fig.add_subplot(221, projection='3d')
    axp.scatter(df['x'], df['z'], df['y'], s=0.05)
    axp.set_xlabel('x')
    axp.set_ylabel('z')
    axp.set_zlabel('y')
    axp.set_title(csvfile)

    axp1 = fig.add_subplot(222, projection='3d')
    axp1.scatter(df['x'], df['z'], df['y'], s=0.05)
    axp1.set_xlabel('x')
    axp1.set_ylabel('z')
    axp1.set_zlabel('y')
    axp1.set_title(csvfile)
    axp1.view_init(0, 90)

    axp2 = fig.add_subplot(223, projection='3d')
    axp2.scatter(df['x'], df['z'], df['y'], s=0.05)
    axp2.set_xlabel('x')
    axp2.set_ylabel('z')
    axp2.set_zlabel('y')
    axp2.set_title(csvfile)
    axp2.view_init(0, 180)

    axp3 = fig.add_subplot(224, projection='3d')
    axp3.scatter(df['x'], df['z'], df['y'], s=0.05)
    axp3.set_xlabel('x')
    axp3.set_ylabel('z')
    axp3.set_zlabel('y')
    axp3.set_title(csvfile)
    axp3.view_init(-90, 0)

    plt.savefig('distance_plot.png')

def pm_plots(df, df_trimmed, csvfile):

    if 'pmra' in df.columns and 'pmdec' in df.columns:

        fig = plt.figure()

        ax = fig.add_subplot(121)
        ax.scatter(df['pmra'], df['pmdec'], s=0.1)
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_title(csvfile)
        ax.set_xlabel("pmra")
        ax.set_ylabel("pmdec")

        ax1 = fig.add_subplot(122)
        ax1.scatter(df_trimmed['pmra'], df_trimmed['pmdec'], s=0.1)
        ax1.set_xlim(-100, 100)
        ax1.set_ylim(-100, 100)
        ax1.set_title(csvfile + " trimmed")
        ax1.set_xlabel("pmra")
        ax1.set_ylabel("pmdec")

        plt.savefig('pm_plots.png')

    else:
        print("Need columns \'pmra\' and \'pmdec\' to plot proper motion diagram.")
        exit()

def hr_plots(df, csvfile):

    if 'g_rp' in df.columns and 'bp_rp' in df.columns and 'phot_g_mean_mag' in df.columns:

        fig = plt.figure()

        ax1 = fig.add_subplot(121)
        ax1.scatter(df['g_rp'], df['phot_g_mean_mag'], s=1)
        ax1.set_xlim(-1, 3)
        ax1.set_ylim(25, 0)
        ax1.set_title(csvfile + ' g-r')
        ax1.set_xlabel("color index (g-rp)")
        ax1.set_ylabel("abs mag (g)")

        ax2 = fig.add_subplot(122)
        ax2.scatter(df['bp_rp'], df['phot_g_mean_mag'], s=1)
        ax2.set_xlim(-1, 5)
        ax2.set_ylim(25, 0)
        ax2.set_title(csvfile + ' b-r')
        ax2.set_xlabel("color index (bp-rp)")
        ax2.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots.png')

    elif 'bp_rp' in df.columns and 'phot_g_mean_mag' in df.columns:

        fig = plt.figure()

        ax2 = fig.add_subplot(111)
        ax2.scatter(df['bp_rp'], df['phot_g_mean_mag'], s=1)
        ax2.set_xlim(-1, 5)
        ax2.set_ylim(25, 0)
        ax2.set_title(csvfile + ' b-r')
        ax2.set_xlabel("color index (bp-rp)")
        ax2.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots.png')

    elif 'g_rp' in df.columns and 'phot_g_mean_mag' in df.columns:

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.scatter(df['g_rp'], df['phot_g_mean_mag'], s=1)
        ax1.set_xlim(-1, 3)
        ax1.set_ylim(25, 0)
        ax1.set_title(csvfile + ' g-r')
        ax1.set_xlabel("color index (g-rp)")
        ax1.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots.png')

    else:
        print("Need column \'phot_g_mean_mag\' and either columns \'g_rp\' or \'bp_rp\' or both to plot HR diagram(s)")
        exit()


def trim_data(df):
    trimmed_df = df.copy(deep=False)

    PARALLAX_S = 1 / 10
    ASTROMETRIC_EXCESS_NOISE_S = 1
    VISIBILITY_PERIODS_S = 5
    PHOT_G_MEAN_MAG_S = 19

    inds_to_drop = []

    for index, row in trimmed_df.iterrows():
        if 'parallax_error' in df.columns and 'parallax' in df.columns \
        and row['parallax_error'] / row['parallax'] >= PARALLAX_S:
            inds_to_drop.append(index)
        elif 'duplicated_source' in df.columns and row['duplicated_source'] == True:
            inds_to_drop.append(index)
        elif 'astrometric_excess_noise' in df.columns and row['astrometric_excess_noise'] >= ASTROMETRIC_EXCESS_NOISE_S:
            inds_to_drop.append(index)
        elif 'visibility_periods_used' in df.columns and row['visibility_periods_used'] <= VISIBILITY_PERIODS_S:
            inds_to_drop.append(index)
        elif 'phot_g_mean_mag' in df.columns and row['phot_g_mean_mag'] > PHOT_G_MEAN_MAG_S:
            inds_to_drop.append(index)

    trimmed_df.drop(inds_to_drop, inplace=True)

    return trimmed_df


def trimmed_hr(trimmed_df, csvfile):

    if 'g_rp' in trimmed_df.columns and \
                    'bp_rp' in trimmed_df.columns and \
                    'phot_g_mean_mag' in trimmed_df.columns:

        fig = plt.figure()

        ax1 = fig.add_subplot(121)
        ax1.scatter(trimmed_df['g_rp'], trimmed_df['phot_g_mean_mag'], s=1)
        ax1.set_xlim(-1, 3)
        ax1.set_ylim(25, 0)
        ax1.set_title(csvfile + ' trimmed g-r')
        ax1.set_xlabel("color index (g-rp)")
        ax1.set_ylabel("abs mag (g)")

        ax2 = fig.add_subplot(122)
        ax2.scatter(trimmed_df['bp_rp'], trimmed_df['phot_g_mean_mag'], s=1)
        ax2.set_xlim(-1, 5)
        ax2.set_ylim(25, 0)
        ax2.set_title(csvfile + ' trimmed b-r')
        ax2.set_xlabel("color index (bp-rp)")
        ax2.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots_trimmed.png')

    elif 'bp_rp' in trimmed_df.columns and 'phot_g_mean_mag' in trimmed_df.columns:

        fig = plt.figure()

        ax2 = fig.add_subplot(111)
        ax2.scatter(trimmed_df['bp_rp'], trimmed_df['phot_g_mean_mag'], s=1)
        ax2.set_xlim(-1, 5)
        ax2.set_ylim(25, 0)
        ax2.set_title(csvfile + ' trimmed b-r')
        ax2.set_xlabel("color index (bp-rp)")
        ax2.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots_trimmed.png')

    elif 'g_rp' in trimmed_df.columns and 'phot_g_mean_mag' in trimmed_df.columns:

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.scatter(trimmed_df['g_rp'], trimmed_df['phot_g_mean_mag'], s=1)
        ax1.set_xlim(-1, 3)
        ax1.set_ylim(25, 0)
        ax1.set_title(csvfile + ' trimmed g-r')
        ax1.set_xlabel("color index (g-rp)")
        ax1.set_ylabel("abs mag (g)")

        plt.savefig('hr_plots_trimmed.png')

    else:
        print("Need column \'phot_g_mean_mag\' and either columns \'g_rp\' or \'bp_rp\' to plot HR diagram(s)")
        exit()


"""
Machine Learning running DBSCAN on the data to find the cluster(s).
"""

def source_id(d):
    df = d[['x', 'y', 'z', 'pmra', 'pmdec', 'phot_g_mean_mag', 'source_id']]
    X = df.to_numpy()

    db = DBSCAN(eps=5, min_samples=150).fit(X)  # HERE
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    data = np.asarray(labels)
    label = pd.DataFrame({'labels': data})
    label.head(10)

    data = np.asarray(core_samples_mask)
    core = pd.DataFrame({'core_samples_mask': data})

    df.reset_index(drop=True, inplace=True)
    label.reset_index(drop=True, inplace=True)
    core.reset_index(drop=True, inplace=True)

    frames = [df, core, label]
    df_all_temp = pd.concat(frames, sort=False, axis=1)

    return df_all_temp


def machine_learning(d):
    df = d[['x', 'y', 'z', 'pmra', 'pmdec', 'phot_g_mean_mag']]
    X = df.to_numpy()

    db = DBSCAN(eps=5, min_samples=150).fit(X)  # HERE
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    data = np.asarray(labels)
    label = pd.DataFrame({'labels': data})
    label.head(10)

    data = np.asarray(core_samples_mask)
    core = pd.DataFrame({'core_samples_mask': data})

    df.reset_index(drop=True, inplace=True)
    label.reset_index(drop=True, inplace=True)
    core.reset_index(drop=True, inplace=True)

    frames = [df, core, label]
    df_all = pd.concat(frames, sort=False, axis=1)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    # Black removed and is used for noise instead.
    unique_labels = set(labels)  # -1, 0, 1

    # gives different colors to the different clusters, and black is for anamolies
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    fig = plt.figure()
    axp = fig.add_subplot(111, projection='3d')

    # for each unique label, plot it with the color that corresponds to it
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        axp.scatter(xy[:, 0], xy[:, 1], facecolor=tuple(col), s=10)

        xy = X[class_member_mask & ~core_samples_mask]
        axp.scatter(xy[:, 0], xy[:, 1], facecolor=tuple(col),
                    edgecolor='k', s=0.05)

    axp.set_xlabel('x')
    axp.set_ylabel('z')
    axp.set_zlabel('y')
    axp.set_title('DBSCAN')
    plt.savefig('DBSCAN.png')

    return df_all, labels, n_clusters, n_noise

def amount(labels, n_clusters, n_noise):
    arr = [0] * n_clusters
    print("Anomaly: ", n_noise)
    for j in range(0, n_clusters):
        for i in labels:
            if i == j:
                arr[j] += 1
        print("Cluster " + str(j + 1) + ": ", arr[j])
    print()

def compare_hr(df_trimmed, df_all, df_all_temp):
    df_temp = df_all_temp[['source_id']]
    frames = [df_temp, df_all]
    df_all_final = pd.concat(frames, sort=False, axis=1)

    # Compare to trimmed HR diagram
    correct = 0
    incorrect = 0
    temp = []
    df_trimmed_id = pd.Series(list(df_trimmed['source_id']))
    for star in df_all_final.itertuples():
        if star.labels == 0:
            if star.source_id in df_trimmed_id.unique():
                temp.append(star.source_id)
                correct += 1
            else:
                incorrect += 1
    print("Correctly clustered: ", correct)
    print("Incorrectly clustered: ", incorrect)
    print("Accuracy: ", correct/(incorrect+correct))


def main():

    print("\nWelcome to Data Mining the Gaia database!\n")
    print("This program takes input of a csv file, "
          "outputs graphs to help you understand your data, \n"
          "and asks if you want to proceed to the Machine Learning portion "
          "and run DBSCAN on your data to find stellar clusters.\n\n"
          "The csv file you enter should contain these columns for the program to run smoothly:\n"
          "parallax, b, l, pmra, pmdec, phot_g_mean_mag, and either bp_rp or g_rp.\n\n"
          "In order to trim the data for better results, these columns are appreciated, but not necessary:\n"
          "parallax_error, duplicated_source, astrometric_excess_noise, and visibility_periods_used.\n\n")

    arg = input("Please enter a csv file: ")
    #while arg.split(".")[1] != 'csv':
    while 'csv' not in arg:
        arg = input("Please enter a csv file: ")
    csvfile = arg

    # Data Mining
    # Create the df
    df = create_df(csvfile)
    # Plot the data
    distance_plot(df, csvfile)
    hr_plots(df, csvfile)
    # Trim the data
    trimmed_df = trim_data(df)
    # Plot the trimmed data
    trimmed_hr(trimmed_df, csvfile)
    # Plot the proper motion
    pm_plots(df, trimmed_df, csvfile)

    ans = input("Would you like to continue onto DBSCAN? (y/n) ")
    temp = ans
    while temp not in ('y', 'n'):
        ans = input("Please input y or n: ")
        temp = ans
    if temp == 'y':
        print()
        #Machine Learning
        # Find source_id in df
        df_all_temp = source_id(df)
        # Find df for DBSCAN, and array of labels
        df_all, labels, n_clusters, n_noise = machine_learning(df)
        # Calculate how many stars are in the cluster(s) vs anomaly
        amount(labels, n_clusters, n_noise)
        compare_hr(trimmed_df, df_all, df_all_temp)
    else:
        print("Program end.")


if __name__ == '__main__':
    main()