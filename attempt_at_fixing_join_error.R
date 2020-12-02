
train <- read_csv(here("data","train.csv"),
                  col_types = cols(.default = col_guess(), 
                                   calc_admn_cd = col_character()))  %>% 
  select(-classification)

or_schools <- readxl::read_xlsx(here("data", "fallmembershipreport_20192020.xlsx"),
                                sheet = 4) 


ethnicities <- or_schools %>% 
  select(attnd_schl_inst_id = `Attending School ID`,
         attnd_dist_inst_id = `Attending District Institution ID`, #included this to join by district along with school id
         sch_name = `School Name`,
         contains('%')) %>%
  janitor::clean_names()
names(ethnicities) <- gsub('x2019_20_percent', 'p', names(ethnicities))


staff <- import(here("data","staff.csv"),
                setclass = "tbl_df") %>% 
  janitor::clean_names() %>%
  filter(st == "OR") %>%
  select(ncessch,schid,teachers) %>%
  mutate(ncessch = as.double(ncessch))


frl <- import(here("data","frl.csv"),
              setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))


stu_counts <- import(here("data","achievement-gaps-geocoded.csv"),
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))



frl_stu <- left_join(frl, stu_counts)
frl1 <- left_join(frl_stu, staff)

frl <- frl1 %>% 
  mutate(fl_prop = free_lunch_qualified/n,
         rl_prop = reduced_price_lunch_qualified/n) %>%
  select(ncessch,fl_prop, rl_prop, teachers)


train_frl <- left_join(train, frl)
d <- left_join(train_frl, ethnicities)


