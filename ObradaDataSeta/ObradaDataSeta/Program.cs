using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObradaDataSeta
{
    class Program
    {
        static void Main(string[] args)
        {

            string[] lines = File.ReadAllLines(@"./../../data/shot_logs.csv");
            lines = lines.Skip(1).ToArray();
            DefensiveRating defensiveRating = new DefensiveRating();            
            List<List<string>> newLines = new List<List<string>>();
            
            foreach(string line in lines)
            {
                
                string[] parameters = line.Split(',');
                bool exists = new bool();
                for(int i=newLines.Count-1; i>=0; i--)
                {
                    exists = false;
                    List<string> chek = newLines[i];
                    
                    if (chek.Contains(parameters[0]) && chek.Contains(parameters[parameters.Length - 2]))
                    {                        
                        chek[8] = (int.Parse(chek[8])+ int.Parse(parameters[parameters.Length - 3])).ToString();
                        exists = true;
                        break;
                    }
                }
                if (!exists)
                {
                    List<string> newLine = new List<string>();
                    string pera = parameters[2];
                    string teamName = "";
                    teamName+=pera[pera.Length - 4];
                    teamName += pera[pera.Length - 3];
                    teamName += pera[pera.Length - 2];                   

                    newLine.Add(parameters[0]);//game id
                    newLine.Add(parameters[1] + "," + parameters[2]);//teams
                    newLine.Add(parameters[3]); // location H/A
                    newLine.Add(parameters[4]); // win W/L
                    newLine.Add(parameters[5]); // final margin
                    newLine.Add(defensiveRating.getTeamDefensiveRank1415(teamName).ToString()); // defensive rating
                    newLine.Add(parameters[parameters.Length - 2]); //player name
                    newLine.Add(parameters[parameters.Length - 1]); //player id
                    newLine.Add(parameters[parameters.Length - 3].ToString());  //points
                    newLines.Add(newLine);                   
                }                
            }
            foreach (List<string> str in newLines)
            {
                foreach (string stri in str)
                {
                    Console.WriteLine(stri);
                }
            }
            
            string comma = ",";
            string filePath = @"./../../data/dataSet.csv";
            string[] columns = { "GAME_ID", "TEAMS", "LOCATION", "W/L", "FINAL_MARGIN", "DEFENSIVE_RATING", "PLAYER_NAME", "PLAYER_ID", "POINTS" };
            StringBuilder sb = new StringBuilder();
            sb.AppendLine(string.Join(comma, columns));
            File.WriteAllText(filePath, sb.ToString());

            StringBuilder sb1 = new StringBuilder();
            foreach(List<string> str in newLines)
            {
                sb1.AppendLine(string.Join(comma, str.ToArray()));
            }
            File.AppendAllText(filePath, sb1.ToString());
            Console.ReadKey();

        }
    }
}
