library(rsconnect)
# rsconnect::setAccountInfo(name='njpang',
#                           token='4A1B08DE5C3DDDB4D01C9DD2A94AC512',
#                           secret='CSSF67k1U0M4fDWjl+W0TZzXNfhKiWn65qWH5ykn')

rsconnect::deployApp(appDir = getwd(),
                     
                     appFiles=c('app.R',
                                'crime.csv',
                                "disaster.csv",
                                'locpolitics.csv',
                                'natpolitics.csv',
                                "overtime_plot.csv",
                                'politics.csv',
                                'sports.csv',
                                "weather.csv"
                     ),
                     account = 'njpang',
                     server = 'shinyapps.io')


