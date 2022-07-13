/*
SQLyog Enterprise - MySQL GUI v6.56
MySQL - 5.5.5-10.1.13-MariaDB : Database - covid
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/`covid` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `covid`;

/*Table structure for table `covid_reg` */

DROP TABLE IF EXISTS `covid_reg`;

CREATE TABLE `covid_reg` (
  `S.no` int(100) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `Email` varchar(100) DEFAULT NULL,
  `psw` varchar(100) DEFAULT NULL,
  `cpsw` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`S.no`)
) ENGINE=InnoDB AUTO_INCREMENT=11 DEFAULT CHARSET=latin1;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
