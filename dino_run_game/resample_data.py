import os
import csv
from time import time, strftime, gmtime, localtime

label = "down"
id = "3"

def process_file(inputDir, fileName, outputDir):
        with open(os.path.join(inputDir, fileName)) as csvfile:
            line_count = 0
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            start_time = -1.0
            all_data = []

            for row in reader:
                line_count += 1
                if line_count == 1:
       #             print(f'Column names are {", ".join(row)}')
                    continue
                
                if start_time < 0: 
                    start_time = float(row[0])

                processed_data = []
                processed_data.append(float(row[0]) - start_time)
                for i in range(len(row) - 1):
                    processed_data.append(row[i + 1])
                all_data.append(processed_data)
            
            #print(all_data[0:5])

        window_size = 128
        shift_size = 64

        for i in range(0, len(all_data) - window_size, shift_size):
            sampleFileName = "%s_%06d.csv" % (fileName.split('.')[0], i)
            with open(os.path.join(outputDir, sampleFileName), mode='w') as csv_output:
                writer = csv.writer(csv_output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["timestamp","TP9","AF7","AF8","TP10","Right AUX"])
                writer.writerows(all_data[i:i+window_size])


def main():
    dataDirOriginal = os.path.join(os.getcwd(), "data")
    dataDirProcessed = os.path.join(os.getcwd(), "processed_data_" + strftime('%Y-%m-%d-%H.%M.%S', localtime()))

    if not os.path.exists(dataDirOriginal):
        raise RuntimeError('Input data directory does not exist: %s' % (dataDirOriginal))

    if not os.path.exists(dataDirProcessed):
        os.mkdir(dataDirProcessed)

    count = 0
    for f in os.listdir(dataDirOriginal):
        if f.endswith(".csv"):
            count += 1
            print(f"Processing %s." % (f))
            process_file(dataDirOriginal, f, dataDirProcessed)
    print("%d .csv file processed." % (count))


if __name__ == "__main__":
    main()