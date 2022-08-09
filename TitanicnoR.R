setwd("/Users/Gugah/Documents/")
getwd()

# Carregando pacotes
library(dplyr)
# Fornecer os mesmo valores que os meus
set.seed(40) 

# Carregando os dados
treino <- read.csv("train.csv", stringsAsFactors = FALSE)
teste <- read.csv("test.csv", stringsAsFactors = FALSE)

# Visualizando os dados
str(treino)

# Histograma idade
hist(treino$Age, main = "Histograma", ylab = "Idade")

# Verificando quantos Na existe no meu dataset
colSums(is.na(treino))
colSums(is.na(teste))

# Separaremos o treino e o teste pela coluna 1 e 2 la na frente
treino["fold"] = 1
teste["fold"] = 2

# Unindo os dataset
total <- bind_rows(treino, teste)

# Boxplot
boxplot(total$Fare)

# Boxplot vimos alguns valores fora da curva, vamos filtrar eles abaixo de 200
total <- mutate(total, Fare = ifelse(Fare < 200, Fare, NA_real_))

# Embarked
total <- filter(total, Embarked != "")   
total <- total %>%
  mutate(Embarked = as.numeric(as.factor(Embarked)),
         Sex = as.numeric(as.factor(Sex)),
         Ticket = as.numeric(as.factor(Ticket)))

# Observamos pelo View que a coluna Cabin é praticamente toda NULA,
# então vamos remover por completo
total$Cabin <- NULL

# A Coluna Name também não vai ter significancia para nosso modelo
# Vamos remover também
total$Name <- NULL

# Verificando se tem valor NaN no meu dataset
colSums(is.na(total))

# Agora iremos fazer a transformação dos valores NaN de Age e Fare pela media
total$Age[is.na(total$Age)] <- mean(total$Age, na.rm = TRUE)
total$Fare[is.na(total$Fare)] <- mean(total$Fare, na.rm = TRUE)

# Arrendodar os valores 
total$Age <- round(total$Age, 0)

# Verificando se tem valor NaN no meu dataset
colSums(is.na(total))

# Como eliminamos todas os NaN, podemos separar os dataset novamente
# E remover a coluna fold e PassengerID do Treino, fold e Survived do teste 
treino <- filter(total, fold == 1)
teste <- filter(total, fold == 2)
treino$fold <- NULL
teste$fold <- NULL
teste$Survived <- NULL
treino$PassengerId <- NULL

# Removendo o dataset Total
rm(total)

# Criando Validação do treino
1:nrow(treino)
prop <- trunc(nrow(treino) * 0.8)
mask <- sample(1:nrow(treino), prop)
treino2 <- treino[mask,]
validacao <- treino[-mask,]

# Carregando a biblioteca 
library(e1071)

# Treinando o modelo Support Vector Machine (SVM)
modelo_svm_v1 <- svm(Survived ~ ., 
                     data = treino2, 
                     type = 'C-classification', 
                     kernel = 'radial') 

# Previsões nos dados de treino
pred_train <- predict(modelo_svm_v1, treino2) 

# Percentual de previsões corretas com dataset de treino
mean(pred_train == treino2$Survived) 

# Previsões nos dados de validação
pred_val <- predict(modelo_svm_v1, validacao)

# Percentual de previsões corretas com dataset de validação
mean(pred_val == validacao$Survived)

# Confusion Matrix
table(pred_val, validacao$Survived)

# Usando o KNN
# Carregando o pacote Class
library(class)

# Criando os labels para os dados de treino e de teste
dados_treino_labels <- treino[mask, 1]
dados_teste_labels <- treino[-mask, 1]

# Carregando o gmodels
library(gmodels)

# Usando a função scale() para padronizar o z-score 
dados_z <- as.data.frame(scale(treino))

# Confirmando transformação realizada com sucesso
summary(dados_z$area_mean)

# Criando um novo dataset de treino e de teste
treino3 <- dados_z[mask,]
validacao2 <- dados_z[-mask,]

# Classificando
modelo_knn <- knn(train = treino3, 
                  test = validacao2,
                  cl = dados_treino_labels, 
                  k = 5)

# Criando uma tabela cruzada dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo_knn, prop.chisq = FALSE)

# Percentual de previsões corretas com dataset de validação
mean(modelo_knn == treino3$Survived)

# Confusion Matrix
table(modelo_knn, validacao$Survived)

# Previsões nos dados de teste
pred_test <- predict(modelo_svm_v1, select(teste, -PassengerId))

# Criamos um dataset para salvar nosso teste
df <- data.frame(PassengerId = teste$PassengerId, Survived = as.integer(pred_test))
df <- mutate(df, Survived = ifelse(Survived == 1, 0, 1))
df <- df %>%
  arrange(PassengerId)

# Salvamos nosso modelo em arquivo csv 
write.csv(df, "/Users/Gugah/Documents/submission.csv", row.names = FALSE)