#Saving the data in a variable
motion_data <- read.csv("motion_stats.csv")
print(motion_data)

#To check where it is looking for the document
getwd()

# Opening the package to create the plots
library(ggplot2)

#Box plot to summarize how much motion was detected over time for each heart type
#can see it was concentrated aound 0 for abnormal since the heart didnt contract as widely as a healthy heart should
ggplot(motion_data, aes(x = type, y = mean_motion)) + 
  geom_boxplot()

#Graph to visualize motion per frame
ggplot(motion_data, aes(x=  frame, y = mean_motion, color = type)) +
  geom_line()

# Average mean motion for each type
aggregate(mean_motion ~ type, data = motion_data, mean)

# Tells us how spatially variable the motion is i.e. if some pixels move more than others in the same frame
aggregate(std_motion ~ type, data = motion_data, mean)

# To see if the mean motion values are statistically different
t.test(mean_motion ~ type, data = motion_data)
