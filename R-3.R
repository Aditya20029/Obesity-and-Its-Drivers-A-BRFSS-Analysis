library(tidyverse)
library(car)
library(ggplot2)

data <- read.csv("C:\\Users\\Adity\\Downloads\\Cleaned_Data.csv")

data <- data %>%
  mutate(urban_rural = ifelse(latitude > 60, "Rural", "Urban"))

obesity_data <- data %>% filter(topic == "Obesity / Weight Status")
physical_activity_data <- data %>% filter(topic == "Physical Activity - Behavior")

# Obesity Summary
obesity_summary <- obesity_data %>%
  group_by(urban_rural) %>%
  summarise(mean_obesity = mean(data_value, na.rm = TRUE),
            sd_obesity = sd(data_value, na.rm = TRUE),
            n = n())

print("Obesity Summary by Urban/Rural")
print(obesity_summary)

# Physical Activity Summary
physical_activity_summary <- physical_activity_data %>%
  group_by(urban_rural) %>%
  summarise(mean_activity = mean(data_value, na.rm = TRUE),
            sd_activity = sd(data_value, na.rm = TRUE),
            n = n())

print("Physical Activity Summary by Urban/Rural")
print(physical_activity_summary)

# Obesity Boxplot
ggplot(obesity_data, aes(x = urban_rural, y = data_value)) +
  geom_boxplot() +
  ggtitle("Obesity Prevalence by Urban/Rural") +
  xlab("Urban/Rural") +
  ylab("Obesity Prevalence (%)")

# Physical Activity Boxplot
ggplot(physical_activity_data, aes(x = urban_rural, y = data_value)) +
  geom_boxplot() +
  ggtitle("Physical Activity by Urban/Rural") +
  xlab("Urban/Rural") +
  ylab("Physical Activity (%)")

# Levene's Test for Obesity
levene_obesity <- leveneTest(data_value ~ urban_rural, data = obesity_data)
print("Levene's Test for Obesity Variance:")
print(levene_obesity)

# Levene's Test for Physical Activity
levene_activity <- leveneTest(data_value ~ urban_rural, data = physical_activity_data)
print("Levene's Test for Physical Activity Variance:")
print(levene_activity)

combined_data <- obesity_data %>%
  select(locationdesc, urban_rural, age_years, income, education, datasource, latitude, longitude, data_value) %>%
  rename(obesity_prevalence = data_value) %>%
  left_join(physical_activity_data %>%
              select(locationdesc, urban_rural, data_value) %>%
              rename(physical_activity = data_value), 
            by = c("locationdesc", "urban_rural"))

# Remove rows with missing values
combined_data <- combined_data %>% filter(!is.na(obesity_prevalence), !is.na(physical_activity))

# Linear Regression: Obesity Prevalence ~ Physical Activity + Urban/Rural
lm_model <- lm(obesity_prevalence ~ physical_activity + urban_rural, data = combined_data)
print("Linear Model Summary:")
print(summary(lm_model))

# Interaction Model: Physical Activity * Urban/Rural
interaction_model <- lm(obesity_prevalence ~ physical_activity * urban_rural, data = combined_data)
print("Interaction Model Summary:")
print(summary(interaction_model))

ggplot(combined_data, aes(x = physical_activity, y = obesity_prevalence, color = urban_rural)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ggtitle("Obesity Prevalence vs. Physical Activity by Urban/Rural") +
  xlab("Physical Activity (%)") +
  ylab("Obesity Prevalence (%)")

# Variance Test for Obesity Prevalence
var_obesity <- var.test(obesity_prevalence ~ urban_rural, data = combined_data)
print("Variance Test for Obesity Prevalence:")
print(var_obesity)

# Variance Test for Physical Activity
var_activity <- var.test(physical_activity ~ urban_rural, data = combined_data)
print("Variance Test for Physical Activity:")
print(var_activity)

