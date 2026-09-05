/* Live retro team cards. Pure data helpers are also exercised under Node. */
(function (root) {
  "use strict";
  const families = [
    "duck tiger bulldog eagle wildcat bear horse bull ram owl wolf rooster panther bison husky cardinal",
    "alligator beaver badger wolverine gopher turtle frog roadrunner turkey insect elephant kangaroo lion dragon buckeye orange",
    "knight pirate cowboy mountaineer miner spartan leprechaun sailor devil tree blob wave hound ibis jay player"
  ].map(s => s.split(" "));
  const overrides = {
    "Akron":"kangaroo", "Alabama":"elephant", "Cincinnati":"wildcat", "Coastal Carolina":"rooster",
    "Delaware":"rooster", "Georgia Tech":"insect", "Iowa":"eagle", "Iowa State":"cardinal",
    "James Madison":"bulldog", "Kansas":"jay", "Kent State":"eagle", "Liberty":"eagle",
    "Miami":"ibis", "Miami (OH)":"eagle", "Michigan":"wolverine", "Middle Tennessee":"horse", "Navy":"ram",
    "North Carolina":"ram", "North Texas":"eagle", "Notre Dame":"leprechaun", "Ohio":"wildcat",
    "Ohio State":"buckeye", "Old Dominion":"lion", "Oregon State":"beaver", "Penn State":"lion",
    "Purdue":"player", "Sacramento State":"insect", "Sam Houston":"wildcat", "Stanford":"tree",
    "Syracuse":"orange", "TCU":"frog", "Tennessee":"hound", "Texas":"bull", "Texas A&M":"hound",
    "Texas State":"wildcat", "Texas Tech":"cowboy", "Tulane":"wave", "Tulsa":"insect",
    "UAB":"dragon", "UL Monroe":"eagle", "UNLV":"cowboy", "Utah":"eagle", "UTSA":"roadrunner",
    "Vanderbilt":"sailor", "Virginia":"knight", "Virginia Tech":"turkey", "Wake Forest":"devil",
    "Western Kentucky":"blob", "Charlotte":"miner", "New Mexico State":"cowboy", "Oklahoma":"horse"
  };
  function mascot(team, metadata) {
    const name = (metadata.mascot || "").toLowerCase();
    const rules = [[/falcon|eagle|hawk/,"eagle"],[/tiger/,"tiger"],[/bulldog/,"bulldog"],
      [/wildcat|bearkat/,"wildcat"],[/bear|bruin/,"bear"],[/bronco|mustang/,"horse"],
      [/bull/,"bull"],[/ram/,"ram"],[/owl/,"owl"],[/wolv(es)?|wolf|lobo/,"wolf"],
      [/gamecock/,"rooster"],[/panther|cougar|jaguar/,"panther"],[/buffalo|bison|herd/,"bison"],
      [/husky|huskies/,"husky"],[/cardinals/,"cardinal"],[/gator/,"alligator"],
      [/badger/,"badger"],[/wolverine/,"wolverine"],[/gopher/,"gopher"],[/terrapin/,"turtle"],
      [/duck/,"duck"],[/knight/,"knight"],[/pirate/,"pirate"],[/cowboy/,"cowboy"],
      [/mountaineer/,"mountaineer"],[/miner|49er/,"miner"],[/spartan|trojan/,"spartan"],[/devil/,"devil"]];
    const family = overrides[team] || (rules.find(([re]) => re.test(name)) || [null,"player"])[1];
    const sheet = families.findIndex(a => a.includes(family)), cell = families[sheet].indexOf(family);
    return { sheet, cell, family };
  }
  function data(team, ratings, schedule) {
    const points = (ratings.history || []).flatMap(s => {
      const r = (s.teams || []).find(r => r.team === team);
      return r && Number.isFinite(r.power) ? [{...r, week:s.week, label:s.label, baseline:s.label === "Preseason"}] : [];
    });
    if (!points.length) {
      const r = (ratings.teams || []).find(r => r.team === team);
      if (r) points.push({...r, label:"Current", baseline:false});
    }
    const games = schedule.filter(g => g.f && g.hp != null && g.ap != null && (g.h === team || g.a === team))
      .sort((a,b) => (a.w-b.w) || String(a.d).localeCompare(String(b.d))).map(g => {
        const home = g.h === team;
        const event = (ratings.game_history || []).find(e => e.home === g.h && e.away === g.a && e.week === g.w &&
          (g.id == null || e.id == null || e.id === g.id));
        const scored = home ? g.hp : g.ap, allowed = home ? g.ap : g.hp;
        return {week:g.w, opponent:home ? g.a : g.h, site:g.n ? "vs" : home ? "vs" : "at",
          scored, allowed, result:scored > allowed ? "W" : scored < allowed ? "L" : "T",
          probability:event && Number.isFinite(event.p_home) ? (home ? event.p_home : 1-event.p_home) : null,
          delta:event && Number.isFinite(event.home_rating_delta) ? (home ? event.home_rating_delta : -event.home_rating_delta) : null};
      });
    const first=points[0], last=points[points.length-1];
    return {points, games, first, last, rankChange:first && first.baseline ? first.rank-last.rank : null,
      wins:games.filter(g=>g.result==="W").length, losses:games.filter(g=>g.result==="L").length,
      ties:games.filter(g=>g.result==="T").length};
  }
  root.TeamCard = {mascot, data};
  if (typeof module !== "undefined") module.exports = root.TeamCard;
})(typeof window !== "undefined" ? window : globalThis);
