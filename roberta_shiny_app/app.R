# Packages
if(!require('pacman')){install.packages('pacman')}
pacman::p_load(shiny, tidyverse, data.table, shinyWidgets)

# Read in data ------------------------------------------------------------
# plot 1 data
plot1_data <- fread("overtime_plot.csv", header = TRUE)
# Politics (all)
data_pol <- read_csv("politics.csv")
# National politics
data_polnat <- read_csv("natpolitics.csv")
# Local Politics
data_polloc <- read_csv("locpolitics.csv")
# Crime 
data_crime <- read_csv("crime.csv")
# weather
data_weather <- read_csv("weather.csv")
# sports
data_sports <- read_csv("sports.csv")
# Disaster
data_disaster <- read_csv("disaster.csv")

# Plot 1 Function ---------------------------------------------------------
plot_overtime <- function(topic_ls, year_ls, city_ls){
  plot <- plot1_data %>% 
    filter(topic %in% topic_ls,
           year %in% year_ls,
           city_state %in% city_ls) %>%
    group_by(date) %>% 
    arrange(date, topic) %>% 
    mutate(hi = cumsum(smoothed_mean_within_day),
           lo = lag(hi) %>% replace_na(., 0)) %>%  
    ungroup %>% 
    ggplot(aes(x = date, fill = topic, ymax=smoothed_mean_within_day, ymin =0, alpha = .01, color = topic))+
    geom_ribbon()+
    ylab("Minutes per 30-minute episode")+
    xlab("")+ 
    facet_wrap(facets = "topic", nrow = 2)+
    theme_bw()+
    theme(legend.position = "none")+
    ggtitle("Frequency of Topics in Local News Coverage")
  
  return(plot)
}
# Plot 2 Function ---------------------------------------------------------
calendar_plot <- function(title, fill, data, year_ls, city_ls) {
  plot <- data %>% 
    mutate(monthf = factor(monthf, levels = c("Jan.",
                                              "Feb.",
                                              "March",
                                              "April",
                                              "May",
                                              "June",
                                              "July",
                                              "Aug.",
                                              "Sept.",
                                              "Oct.",
                                              "Nov.",
                                              "Dec.")),
           weekday = factor(weekday, levels = c("Monday",
                                                "Tuesday",
                                                "Wednesday",
                                                "Thursday",
                                                "Friday",
                                                "Saturday",
                                                "Sunday")),
           weekday = fct_rev(weekday)) %>% 
    filter(year %in% year_ls,
           city_state %in% city_ls) %>% 
    ggplot(aes(monthweek, weekday, fill = topic_prop)) +
    geom_tile(colour = "white") +
    facet_grid(year ~ monthf) +
    scale_fill_gradient(low = "green", high = "red") +
    labs(
      x = "Week of Month",
      y = "",
      title = title,
      fill = fill
    )
  
  return(plot)
}

# Create function to add line breaks to shiny app
linebreaks <- function(n){HTML(strrep(br(), n))}

# UI ----------------------------------------------------------------------

ui <- fluidPage(
  titlePanel("Classifying Local News Television Transcripts Using RoBERTa"),
  
  sidebarLayout(
    sidebarPanel(
      helpText('Examine what local news networks cover over time.'),
      # inputs for plot type
      selectInput(inputId = "plot_type",
                  selected = "Density Plot",
                  label = "Select a type of plot:",
                  choices = c("Density Plot", "Calendar Plot")),
      # Inputs for topic - pickerInput to allow multiple selections
      pickerInput(inputId = "topic",
                  selected = "Politics",
                  label = "Select a subset of topics or all topics:",
                  choices = c("Politics", "National Politics", "Local Politics",
                              "Crime", "Disaster", "Sports", "Weather"),
                  options = list(`actions-box` = TRUE), multiple = T),
      # inputs for year
      pickerInput(inputId = "year",
                  selected = "2016",
                  label = "Select a subset of years or all years:",
                  choices = c("2014", "2015", "2016", "2017", "2018"),
                  options = list(`actions-box` = TRUE), multiple = T),
      textOutput("sidenote1"),
      linebreaks(1),
      uiOutput("tab")
      
  ),
    mainPanel(
      plotOutput("plot"),
      textOutput("note1"),
      linebreaks(1),
      textOutput("note2"),
      textOutput("note3"),
      textOutput("note4"),
      textOutput("note5")
    )
  )
)

# Server ----------------------------------------------------------------------

server <- function(input, output, session){
  ########## A) Messages
  url <- a("at this link", href="https://github.com/npangakis")
  output$tab <- renderUI({
    tagList("View full project on Github", url)
  })
  output$sidenote1 <- renderText({
    print("Note: This is joint work with Sam Wolken and Chloe Ahn")
  })
  output$note1 <- renderText({
    req(input$topic, input$year)
    print("Note: Total number of collected transcripts is 17,732 across Philadelphia, Boston, and New York City local news broadcasts (2014-2018).")
  })
  output$note2 <- renderText({
    req(input$topic, input$year)
    topic_ls <- input$topic
    if ("Politics" %in% topic_ls | "National Politics" %in% topic_ls | "Local Politics" %in% topic_ls | "Crime" %in% topic_ls | "Sports" %in% topic_ls | "Weather" %in% topic_ls){
      print("Interesting Observations:")
    }
  })
  output$note3 <- renderText({
    req(input$topic, input$year)
    topic_ls <- input$topic
    if ("Politics" %in% topic_ls | "National Politics" %in% topic_ls | "Local Politics" %in% topic_ls) {
      print("---Coverage of national politics increases around the 2016 election with three distinct spikes that are directly attributable to: a) Donald Trump becoming the Republican nominee in July 2016; b) the November general election; and c) the January Inauguration. Local politics, however, does not change significantly over time.")
    }
    })
  output$note4 <- renderText({
    req(input$topic, input$year)
    topic_ls <- input$topic
    if ("Crime" %in% topic_ls | "Weather" %in% topic_ls) {
      print("---Interestingly, crime and weather are by far the most popular topics. The calender plots provide suggestive evidence that crime coverage is fairly consistent and that weather coverage may increase in the winter months of January and February.") 
    }
  })
  output$note5 <- renderText({
    req(input$topic, input$year)
    topic_ls <- input$topic
    if ("Sports" %in% topic_ls) {
      print("---There is a dramatic increase in sports coverage in 2018, which is likely driven by the Philadelphia Eagles winning the 2018 Super Bowl (since the 2018 data is only from Philadelphia).") 
    }
  })
  
  ########## B) Update inputs based on user selection
  observe({
    # observe plot_type input
    plot_type <- input$plot_type
    
    # based on plot_type input, filter input topic options to only include a single topic
    if (plot_type == "Calendar Plot") {
      updatePickerInput(session,
                        inputId = "topic",
                        selected = "Politics",
                        label = "Select a news topic:",
                        choices = c("Politics", "National Politics", "Local Politics", 
                                    "Crime", "Disaster", "Sports", "Weather"),
                        options = list(`actions-box` = FALSE,
                                       "max-options" = 1,
                                       "max-options-text" = "For calendar plot, you can only select one topic"))
    } else if (plot_type == "Density Plot") {
      updatePickerInput(session,
                        inputId = "topic",
                        selected = "Politics",
                        label = "Select a subset of topics or all topics:",
                        choices = c("Politics", "National Politics", "Local Politics",
                                    "Crime", "Disaster", "Sports", "Weather"),
                        options = list(`actions-box` = TRUE,
                                       "max-options" = 99))
    }
  }) 

  ########## C) Plots
  output$plot <- renderPlot({
    req(input$topic, input$year)
    topic_ls <- input$topic
    year_ls <- input$year
    city_ls <- c("Boston, MA", "New York, NY", "Philadelphia, PA")
      
    if (input$plot_type == "Density Plot"){
      plot_overtime(topic_ls, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Politics"){
      fill = "% of Broadcasts Discussing Politics"
      title = "Time-Series Politics Heatmap"
      calendar_plot(title, fill, data_pol, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "National Politics"){
      fill = "% of Broadcasts Discussing National Politics"
      title = "Time-Series National Politics Heatmap"
      calendar_plot(title, fill, data_polnat, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Local Politics"){
      fill = "% of Broadcasts Discussing Local Politics"
      title = "Time-Series Local Politics Heatmap"
      calendar_plot(title, fill, data_polloc, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Crime"){
      fill = "% of Broadcasts Discussing Crime"
      title = "Time-Series Crime Heatmap"
      calendar_plot(title, fill, data_crime, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Disaster"){
      fill = "% of Broadcasts Discussing Disaster"
      title = "Time-Series Disaster Heatmap"
      calendar_plot(title, fill, data_disaster, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Sports"){
      fill = "% of Broadcasts Discussing Sports"
      title = "Time-Series Sports Heatmap"
      calendar_plot(title, fill, data_sports, year_ls, city_ls)
    } else if (input$plot_type == "Calendar Plot" & topic_ls == "Weather"){
      fill = "% of Broadcasts Discussing Weather"
      title = "Time-Series Weather Heatmap"
      calendar_plot(title, fill, data_weather, year_ls, city_ls)
    }
    
  })

}

# Run the Application -----------------------------------------------------
shinyApp(ui = ui, server = server)

