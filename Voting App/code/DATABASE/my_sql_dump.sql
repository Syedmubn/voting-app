/*!40101 SET NAMES utf8 */;
/*!40101 SET SQL_MODE=''*/;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `voting_system` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `voting_system`;

/* Drop existing tables if they exist */
DROP TABLE IF EXISTS `candidates`;
DROP TABLE IF EXISTS `election_schedule`;
DROP TABLE IF EXISTS `voters`;
DROP TABLE IF EXISTS `vote`;
DROP TABLE IF EXISTS `student`;

/* Create the candidates table */
CREATE TABLE `candidates` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `position` varchar(100) NOT NULL,
  `member_name` varchar(100) NOT NULL,
  `party_name` varchar(100) DEFAULT NULL,
  `picture` blob,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/* Create the election schedule table */
CREATE TABLE `election_schedule` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `start_time` datetime NOT NULL,
  `end_time` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


/* Create the votes table */
CREATE TABLE `vote` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `cnic` varchar(100) NOT NULL,
  `position` varchar(100) NOT NULL,
  `vote` varchar(100) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/* Create the student table */
CREATE TABLE `student` (
  `id` int(10) NOT NULL AUTO_INCREMENT,
  `first_name` varchar(100) DEFAULT NULL,
  `middle_name` varchar(100) DEFAULT NULL,
  `last_name` varchar(100) DEFAULT NULL,
  `cnic` varchar(100) DEFAULT NULL UNIQUE,
  `voter_id` varchar(100) DEFAULT NULL UNIQUE,
  `email` varchar(100) DEFAULT NULL,
  `phone_number` varchar(100) DEFAULT NULL, /* Added phone number */
  `department` varchar(100) DEFAULT NULL, /* Added department */
  `semester` varchar(100) DEFAULT NULL, /* Added semester */
  `photo` blob, /* Added to store photo filename */
  `face_embedding` blob, /* Added to store photo filename */
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
