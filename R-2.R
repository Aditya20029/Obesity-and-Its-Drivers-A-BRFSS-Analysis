library(tidyverse)

data <- read.csv("C:\\Users\\Adity\\Downloads\\Cleaned_Data.csv", stringsAsFactors = FALSE)

cleaned_data <- data %>%
  filter(!is.na(data_value), !is.na(income), !is.na(education)) %>%
  filter(topic == "Obesity / Weight Status", 
         grepl("obesity", question, ignore.case = TRUE))

cleaned_data$income <- as.factor(cleaned_data$income)
cleaned_data$education <- as.factor(cleaned_data$education)

# Univariate Analysis: 
ggplot(cleaned_data, aes(x = income, y = data_value)) +
  geom_boxplot(fill = "skyblue") +
  labs(
    title = "Obesity Levels by Income Groups",
    x = "Income Groups",
    y = "Obesity Levels (%)"
  ) +
  theme_minimal()

ggplot(cleaned_data, aes(x = education, y = data_value)) +
  geom_boxplot(fill = "orange") +
  labs(
    title = "Obesity Levels by Education Levels",
    x = "Education Levels",
    y = "Obesity Levels (%)"
  ) +
  theme_minimal()

# Multivariate Analysis: Linear Regression

model <- lm(data_value ~ income + education, data = cleaned_data)


summary(model)


interaction_model <- lm(data_value ~ income * education, data = cleaned_data)

summary(interaction_model)

ggplot(cleaned_data, aes(x = income, y = data_value, color = education)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    title = "Interaction between Income and Education on Obesity Levels",
    x = "Income Groups",
    y = "Obesity Levels (%)",
    color = "Education Level"
  ) +
  theme_minimal()
