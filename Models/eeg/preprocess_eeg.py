import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

def movingWindow(dataList):
    # Define the window size and stride
    window_size = 64
    stride = 64

    # Initialize an empty list to store the windows
    windows_list = []

    # Iterate over the column data with the given stride
    for start in range(0, len(dataList), stride):
        # Define the end of the window
        end = start + window_size

        if end > len(dataList):
            break

        # Slice the window and add it to the list
        window = dataList[start:end]
        windows_list.append(window)

    return windows_list


def buildData(csv_file_path, labelId):
    # Use pandas to read the CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path, na_values=['NA', 'NaN', ''])

        # Print the first few rows of the DataFrame
        print(df.head())
    except FileNotFoundError:
        print(f"The file {csv_file_path} does not exist.")
    except pd.errors.EmptyDataError:
        print(f"The file {csv_file_path} is empty.")
    except pd.errors.ParserError:
        print(f"The file {csv_file_path} does not appear to be in CSV format.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Define the columns you want to copy
    columns_to_copy = ['RAW_TP9', 'RAW_TP10', 'RAW_AF7', 'RAW_AF8']
    # Create a new DataFrame with the selected columns
    new_df = df[columns_to_copy].copy()

    # Apply pd.to_numeric to the entire DataFrame
    # errors='coerce' will replace non-convertible values with NaN
    df_numeric = new_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the data and transform it
    df_standardized = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns)


    tp9  = df_standardized['RAW_TP9'].tolist()
    tp10 = df_standardized['RAW_TP10'].tolist()
    af7  = df_standardized['RAW_AF7'].tolist()
    af8  = df_standardized['RAW_AF8'].tolist()


    windows_list_tp9 = movingWindow(tp9)
    windows_list_tp10 = movingWindow(tp10)
    windows_list_af7 = movingWindow(af7)
    windows_list_af8 = movingWindow(af8)


    array1 = np.array(windows_list_tp9)
    array2 = np.array(windows_list_tp10)
    array3 = np.array(windows_list_af7)
    array4 = np.array(windows_list_af8)

    X = np.stack((array1, array2, array3, array4), axis=2)

    # Create a 3D array filled with zeros
    y = np.full(X.shape[0], labelId)


    print("Shape of the X array:", X.shape)
    print("Shape of the y array:", y.shape)

    return X, y

def getData():
    # Define the path to your CSV file
    X1, y1 = buildData('./data/up.csv', 1)
    X2, y2 = buildData('./data/down.csv', 2)
    X3, y3 = buildData('./data/left.csv', 3)
    X4, y4 = buildData('./data/right.csv', 4)
    X5, y5 = buildData('./data/neutral.csv', 5)

    X = np.concatenate((X1, X2, X3, X4, X5), axis=0)
    y = np.concatenate((y1, y2, y3, y4, y5), axis=0)
    return X, y


# Generate synthetic multivariate time series data
# X = np.random.random((n_samples, n_timesteps, n_features))
# y = np.random.randint(0, n_classes, n_samples)




# Plotting one column against another using a simple line plot
# plt.figure(figsize=(10, 5))
# plt.plot(df['Delta_TP9'].head(10000), label='delta_tp9')
# plt.title('delta_tp9')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# # Plotting all columns on the same plot
# plt.figure(figsize=(10, 5))
# for column in df.columns:
#     plt.plot(df[column], label=column)
# plt.title('Line Plot of All Columns')
# plt.xlabel('Index')
# plt.ylabel('Value')
# plt.legend()
# plt.show()

# Creating subplots for each column
# fig, axs = plt.subplots(len(df.columns), figsize=(10, 5), sharex=True)
# for i, column in enumerate(df.columns):
#     axs[i].plot(df[column])
#     axs[i].set_title(column)
# plt.xlabel('Index')
# plt.show()
