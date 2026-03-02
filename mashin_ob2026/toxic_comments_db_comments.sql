-- MySQL dump 10.13  Distrib 8.0.44, for Win64 (x86_64)
--
-- Host: localhost    Database: toxic_comments_db
-- ------------------------------------------------------
-- Server version	9.5.0

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
SET @@SESSION.SQL_LOG_BIN= 0;

--
-- GTID state at the beginning of the backup 
--

SET @@GLOBAL.GTID_PURGED=/*!80000 '+'*/ 'c86874be-c4c9-11f0-8b4d-98fa9b8ceff6:1-151';

--
-- Table structure for table `comments`
--

DROP TABLE IF EXISTS `comments`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `comments` (
  `id` int NOT NULL AUTO_INCREMENT,
  `text` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `prediction` float NOT NULL,
  `is_toxic` tinyint(1) NOT NULL,
  `timestamp` datetime DEFAULT CURRENT_TIMESTAMP,
  `processed_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_toxic` (`is_toxic`),
  KEY `idx_timestamp` (`timestamp`)
) ENGINE=InnoDB AUTO_INCREMENT=20 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `comments`
--

LOCK TABLES `comments` WRITE;
/*!40000 ALTER TABLE `comments` DISABLE KEYS */;
INSERT INTO `comments` VALUES (1,'я очень злой иди отсюда ',0.476706,0,'2026-02-16 11:30:55','2026-02-16 06:30:55'),(2,'я очень злой!!',0.0765593,0,'2026-02-16 11:31:14','2026-02-16 06:31:14'),(3,'пошел отсюда',0.737872,1,'2026-02-16 11:31:41','2026-02-16 06:31:41'),(4,'быстро убрался отсюда гад',0.831087,1,'2026-02-16 11:32:05','2026-02-16 06:32:05'),(5,'я самый добрый и хороший',0.0715174,0,'2026-02-16 11:32:24','2026-02-16 06:32:24'),(6,'дуб добрый деревянный ',0.779967,1,'2026-02-20 11:40:43','2026-02-20 06:40:43'),(7,'бд',0.739245,1,'2026-02-20 11:41:29','2026-02-20 06:41:29'),(8,'данность ссанность',0.841716,1,'2026-02-20 11:41:45','2026-02-20 06:41:45'),(9,'токсичность',0.739245,1,'2026-02-20 11:41:56','2026-02-20 06:41:56'),(10,'собко артем',0.828839,1,'2026-02-20 11:42:11','2026-02-20 06:42:11'),(11,'валиулка тимурка',0.828839,1,'2026-02-20 11:42:19','2026-02-20 06:42:19'),(12,'усман хуснияров',0.828839,1,'2026-02-20 11:42:24','2026-02-20 06:42:24'),(13,'добри',0.739245,1,'2026-02-20 11:42:28','2026-02-20 06:42:28'),(14,'добрый',0.417741,0,'2026-02-20 11:42:32','2026-02-20 06:42:32'),(15,'шомка бэтка',0.828839,1,'2026-02-20 11:42:50','2026-02-20 06:42:50'),(16,'мастурбэтр',0.739245,1,'2026-02-20 11:42:58','2026-02-20 06:42:58'),(17,'азалия',0.739245,1,'2026-02-20 11:43:04','2026-02-20 06:43:04'),(18,'красивый добрый мальчик',0.380823,0,'2026-02-20 11:44:02','2026-02-20 06:44:02'),(19,'грязная аналка',0.84672,1,'2026-02-20 11:45:21','2026-02-20 06:45:21');
/*!40000 ALTER TABLE `comments` ENABLE KEYS */;
UNLOCK TABLES;
SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2026-03-02 13:27:46
