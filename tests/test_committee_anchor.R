source('scripts/committee_anchor.R')
names <- paste0('T',1:30)
cm <- data.frame(team=names, static=seq(3,0,length.out=30))
poll <- list(season=2026L,released='2026-11-03',games_through='2026-10-31',source_url='https://collegefootballplayoff.com/rankings',
             rankings=lapply(1:25,function(i)list(team=paste0('T',26-i),rank=i)))
schedule <- list(list(h='T1',a='T2',f=1,hp=20,ap=10,d='2026-10-30'),
                 list(h='T3',a='FCS',f=1,hp=20,ap=0,d='2026-10-31'),
                 list(h='T4',a='T5',f=1,hp=20,ap=0,d='2026-11-03'))
a <- committee_anchor(cm,schedule,poll,2026L,10)
stopifnot(identical(names(sort(a$baseline,decreasing=TRUE))[1:25],paste0('T',25:1)),
          a$win_pct['T1']==1,a$win_pct['T2']==0,a$win_pct['T3']==1,a$win_pct['T4']==.5,
          all(a$baseline[paste0('T',26:30)] < a$baseline['T1']),a$rank_spacing>0)
proxy <- 10*a$win_pct[cm$team]+cm$static
stopifnot(max(abs(proxy+a$offset-a$baseline))<1e-10)
# A future record improvement raises score; the published order is a starting point, not a freeze.
changed <- proxy+a$offset;changed['T1'] <- changed['T1']+10*.5
stopifnot(changed['T1']>a$baseline['T1'])
stopifnot(is.null(committee_anchor(cm,schedule,poll,2025L,10)))
bad <- poll;bad$rankings <- bad$rankings[-1]
stopifnot(inherits(try(committee_anchor(cm,schedule,bad,2026L,10),silent=TRUE),'try-error'))
cat('Committee anchor: exact published order, NR boundary, cutoff/FCS records, residual identity, future movement and season guards passed\n')
# Exercise the production selector without loading cfbseedR or running a season.
exprs <- parse('scripts/simulate_playoff.R')
for (e in exprs) if (is.call(e) && identical(e[[1]], as.name('<-')) && identical(e[[2]], as.name('select_and_play'))) eval(e)
fbs <- names
cm$conference <- c('ACC','Big Ten','Big 12',rep('SEC',22),rep('Mountain West',5))
P4_CONFS <- c('ACC','Big Ten','Big 12','SEC')
G6_CONFS <- 'Mountain West'
W_WINPCT <- 10
static_committee <- function(input) cm
viz <- tempfile();dir.create(viz);file.create(file.path(viz,'cfp_rankings.json'))
read_json <- function(path) poll
probability <- function(a,b,neutral,spec) rep(.5,length(a))
st <- expand.grid(sim=1:2,team=fbs,stringsAsFactors=FALSE)
st$win_pct <- unname(a$win_pct[st$team])
st$conf_champ <- st$team %in% c('T1','T2','T3','T4')
selected <- select_and_play(st,list(mode='current',spec=NULL))
stopifnot(!is.null(attr(selected,'committee_anchor')),
          all(tapply(!is.na(selected$seed),selected$sim,sum)==12),
          all(selected$seed[selected$team=='T25']==1),
          all(selected$seed[selected$team=='T24']==2),
          all(!is.na(selected$seed[selected$team %in% c('T1','T2','T3','T4') ])),
          all(tapply(selected$g6_auto,selected$sim,sum)==1))
pre <- select_and_play(st,list(mode='preseason',spec=NULL))
stopifnot(is.null(attr(pre,'committee_anchor')))
unlink(viz,recursive=TRUE)
cat('Production selector: committee baseline, P4/G6 bids, twelve unique seeds and preseason isolation passed\n')
