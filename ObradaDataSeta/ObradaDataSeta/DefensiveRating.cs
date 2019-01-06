using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObradaDataSeta
{
    class DefensiveRating
    {
        public Dictionary<string, double> defensiveRatings1415 { get; set; }
        public DefensiveRating()
        {
            defensiveRatings1415 = new Dictionary<string, double>();
            defensiveRatings1415.Add("TOR", 100.9);
            defensiveRatings1415.Add("BOS", 101.2);
            defensiveRatings1415.Add("BKN", 100.9);
            defensiveRatings1415.Add("PHI", 101);
            defensiveRatings1415.Add("NYK", 101.2);
            defensiveRatings1415.Add("CLE", 98.7);
            defensiveRatings1415.Add("CHI", 97.8);
            defensiveRatings1415.Add("MIL", 97.4);
            defensiveRatings1415.Add("IND", 97);
            defensiveRatings1415.Add("DET", 99.5);
            defensiveRatings1415.Add("ATL", 97.1);
            defensiveRatings1415.Add("WAS", 97.8);
            defensiveRatings1415.Add("MIA", 97.3);
            defensiveRatings1415.Add("CHA", 97.3);
            defensiveRatings1415.Add("ORL", 101.4);
            defensiveRatings1415.Add("POR", 98.6);
            defensiveRatings1415.Add("OKC", 101.8);
            defensiveRatings1415.Add("UTA", 94.9);
            defensiveRatings1415.Add("DEN", 105);
            defensiveRatings1415.Add("MIN", 106.5);
            defensiveRatings1415.Add("GSW", 99.9);
            defensiveRatings1415.Add("LAC", 100.1);
            defensiveRatings1415.Add("PHX", 103.3);
            defensiveRatings1415.Add("SAC", 105);
            defensiveRatings1415.Add("LAL", 105.3);
            defensiveRatings1415.Add("HOU", 100.5);
            defensiveRatings1415.Add("SAS", 97);
            defensiveRatings1415.Add("MEM", 95.1);
            defensiveRatings1415.Add("DAL", 102.3);
            defensiveRatings1415.Add("NOP", 98.6);
        }

        public double getTeamDefensiveRank1415(string name)
        {

            double rating = defensiveRatings1415[name];
            return rating;
            /*int position = 1;
            foreach (var pair in defensiveRatings1415)
            {
                if (rating > pair.Value)
                    position++;
            }

            return position;*/
        }
    }
}
