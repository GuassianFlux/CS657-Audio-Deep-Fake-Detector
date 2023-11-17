import re
import sys
import os
import math

def main(argv):
    directory = argv[0]
    len = 0
    sum = 0
    false_accept = 0
    false_reject = 0
    predictions = 0
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            metricLog = open(directory + '/' + file, "r")
            for line in metricLog:
                if re.search('Prediction', line):
                    predictions += 1
                    words = line.split();
                    prediction = words[3][:-1]
                    actual = words[5][:-1]
                    if (prediction == 'fake' and actual == 'real'):
                        false_reject += 1
                    if (prediction == 'real' and actual == 'fake'):
                        false_accept += 1
                if re.search('Accuracy', line):
                    percent = float(line[9:17])
                    sum += percent
                    len += 1
    rounded_accuracy = round((sum / len),4)
    error_rate = ((false_accept + false_reject) / 2) / predictions
    rounded_error_percent = round((error_rate * 100),4)
    print ('Accuracy: ' + str(rounded_accuracy))
    print ('EER: ' + str(rounded_error_percent))

if __name__ == "__main__":
    main(sys.argv[1:])