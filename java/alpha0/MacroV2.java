package alpha0;

import battlecode.common.*;
import battlecode.world.*;
import java.util.*;

/** Versioned research action abstraction, authored from official rules and controller APIs.
 * All targets use omniscient physical state; every actual effect passes an official can*/
public final class MacroV2 {
    public static final int ACTION_DIM=40;
    public static final String[] NAMES={"hold","north","northeast","east","southeast","south","southwest","west","northwest",
        "build_north","build_northeast","build_east","build_southeast","build_south","build_southwest","build_west","build_northwest",
        "gather_cheese","return_cheese","claim_mine","guard_king","assault_king","hunt_rat","hunt_cat","retreat","king_relocate",
        "spawn_economy","spawn_assault","rat_trap","cat_trap","clear_own_trap","form_king","carry_enemy","rescue_ally",
        "throw_damage","throw_transport","drop_safe","powered_bite","clear_route","cooperate_economy"};
    private static final Direction[] DIRS=Arrays.copyOf(Direction.values(),8);
    private static final int INF=1_000_000_000;

    static final class Context {
        final EngineBridge.Session session;
        final GameWorld world;
        final InternalRobot actor;
        final RobotControllerImpl rc;
        final ArrayList<InternalRobot> robots=new ArrayList<>();
        final int width,height;
        final Team us;
        ArrayList<MapLocation> cheeseTiles,mineTiles;
        Context(EngineBridge.Session session) {
            this.session=session;world=session.world;actor=session.current;rc=actor.getController();us=actor.getTeam();
            width=world.getGameMap().getWidth();height=world.getGameMap().getHeight();
            world.getObjectInfo().eachRobot(r->{robots.add(r);return true;});robots.sort(Comparator.comparingInt(InternalRobot::getID));
        }
        boolean alive() {return world.getObjectInfo().getRobotByID(actor.getID())!=null;}
        boolean on(MapLocation p) {return p.x>=0 && p.y>=0 && p.x<width && p.y<height;}
        InternalRobot nearest(Team team,UnitType type) {
            InternalRobot best=null;int score=INF;
            for(InternalRobot r:robots) if(r!=actor && r.getTeam()==team && (type==null || r.getType()==type) && !r.isBeingThrown()) {
                int d=distance(rc.getLocation(),r.getAllPartLocations());
                if(d<score) {score=d;best=r;}
            }
            return best;
        }
        MapLocation nearestPart(InternalRobot r) {return r==null?null:nearestPoint(rc.getLocation(),Arrays.asList(r.getAllPartLocations()));}
        boolean mobile() {return rc.isMovementReady() || rc.isTurningReady() || rc.isActionReady();}
    }
    static int distance(MapLocation p,MapLocation[] parts) {int d=INF;for(MapLocation q:parts)d=Math.min(d,p.distanceSquaredTo(q));return d;}
    static MapLocation nearestPoint(MapLocation p,List<MapLocation> points) {MapLocation best=null;int d=INF;for(MapLocation q:points)if(p.distanceSquaredTo(q)<d){d=p.distanceSquaredTo(q);best=q;}return best;}
    static ArrayList<MapLocation> terrain(Context c,boolean mines) {
        if(mines && c.mineTiles!=null)return c.mineTiles;
        if(!mines && c.cheeseTiles!=null)return c.cheeseTiles;
        ArrayList<MapLocation> points=new ArrayList<>();
        for(int y=0;y<c.height;y++)for(int x=0;x<c.width;x++){MapLocation p=new MapLocation(x,y);if(mines?c.world.hasCheeseMine(p):c.world.getCheese(p)>0)points.add(p);}
        if(mines)c.mineTiles=points;else c.cheeseTiles=points;
        return points;
    }
    static MapLocation leadingDirt(Context c,Direction direction) {
        for(MapLocation p:c.actor.getAllPartLocations()) {
            MapLocation next=p.add(direction);
            if(c.on(next) && c.world.getDirt(next) && c.rc.canRemoveDirt(next)) return next;
        }
        return null;
    }
    static boolean canBuild(Context c) {for(Direction d:DIRS)if(c.session.buildLocation(c.rc,d)!=null)return true;return false;}
    static boolean canFace(Context c,MapLocation p) {Direction d=c.rc.getLocation().directionTo(p);return d==Direction.CENTER || d==c.rc.getDirection() || c.rc.canTurn(d);}
    static int count(Context c,Team team,UnitType type) {int n=0;for(InternalRobot r:c.robots)if(r.getTeam()==team && r.getType()==type)n++;return n;}

    static List<Integer> legal(EngineBridge.Session session) {
        ArrayList<Integer> ids=new ArrayList<>();if(session.current==null)return ids;ids.add(0);
        Context c=new Context(session);
        if(c.actor.isGrabbedByRobot() || c.actor.isBeingThrown())return ids;
        for(int i=0;i<8;i++) {
            if(c.rc.canMove(DIRS[i]) || leadingDirt(c,DIRS[i])!=null || (c.rc.canTurn(DIRS[i]) && DIRS[i]!=c.rc.getDirection()))ids.add(i+1);
            if(c.session.buildLocation(c.rc,DIRS[i])!=null)ids.add(i+9);
        }
        boolean mobile=c.mobile();
        InternalRobot king=c.nearest(c.us,UnitType.RAT_KING),enemyKing=c.nearest(c.us.opponent(),UnitType.RAT_KING),enemy=c.nearest(c.us.opponent(),null),cat=c.nearest(Team.NEUTRAL,UnitType.CAT);
        if(mobile && !terrain(c,false).isEmpty())ids.add(17);
        if(mobile && king!=null && c.rc.getType()==UnitType.BABY_RAT && c.rc.getRawCheese()>0)ids.add(18);
        if(mobile && !terrain(c,true).isEmpty())ids.add(19);
        if(mobile && king!=null)ids.add(20);
        if(mobile && enemyKing!=null)ids.add(21);
        if(mobile && c.nearest(c.us.opponent(),UnitType.BABY_RAT)!=null)ids.add(22);
        if(mobile && cat!=null)ids.add(23);
        if(mobile && (enemy!=null || cat!=null))ids.add(24);
        if(mobile && c.rc.getType()==UnitType.RAT_KING && (!terrain(c,false).isEmpty() || !terrain(c,true).isEmpty()))ids.add(25);
        if(canBuild(c)){ids.add(26);if(enemyKing!=null)ids.add(27);}
        if(trapLocation(c,false)!=null)ids.add(28);
        if(trapLocation(c,true)!=null)ids.add(29);
        if(removableTrap(c)!=null)ids.add(30);
        if(c.rc.getType()==UnitType.BABY_RAT && mobile && c.world.getTeamInfo().getCheese(c.us)>=GameConstants.RAT_KING_UPGRADE_CHEESE_COST && count(c,c.us,UnitType.BABY_RAT)>=7)ids.add(31);
        if(mobile && c.rc.getType()==UnitType.BABY_RAT && !c.actor.isCarryingRobot() && c.nearest(c.us.opponent(),UnitType.BABY_RAT)!=null)ids.add(32);
        if(mobile && c.rc.getType()==UnitType.BABY_RAT && !c.actor.isCarryingRobot() && rescueTarget(c)!=null)ids.add(33);
        if(c.actor.isCarryingRobot() && c.rc.isActionReady() && throwDirection(c,false)!=null)ids.add(34);
        if(c.actor.isCarryingRobot() && c.rc.isActionReady() && throwDirection(c,true)!=null)ids.add(35);
        if(c.actor.isCarryingRobot() && c.rc.isActionReady())for(Direction d:DIRS)if(c.rc.canDropRat(d)){ids.add(36);break;}
        if(c.rc.isActionReady() && c.rc.getAllCheese()>0 && attackTarget(c,1)!=null)ids.add(37);
        if(mobile && nearestDirt(c)!=null)ids.add(38);
        if(mobile && (king!=null || c.rc.getType()==UnitType.RAT_KING || !terrain(c,false).isEmpty()))ids.add(39);
        return ids;
    }

    static void pickup(Context c) throws GameActionException {
        if(!c.alive())return;
        MapLocation center=c.rc.getLocation();
        for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){MapLocation p=center.translate(dx,dy);if(c.rc.canPickUpCheese(p))c.rc.pickUpCheese(p);}
    }
    static void deposit(Context c) throws GameActionException {
        if(!c.alive() || c.rc.getType()!=UnitType.BABY_RAT || c.rc.getRawCheese()==0 || !c.rc.isActionReady())return;
        for(InternalRobot king:c.robots)if(king.getTeam()==c.us && king.getType()==UnitType.RAT_KING)for(MapLocation p:king.getAllPartLocations()) {
            if(c.rc.getLocation().distanceSquaredTo(p)>GameConstants.CHEESE_TRANSFER_RADIUS_SQUARED)continue;
            if(c.rc.canTransferCheese(p,c.rc.getRawCheese())){c.rc.transferCheese(p,c.rc.getRawCheese());return;}
            Direction d=c.rc.getLocation().directionTo(p);
            if(c.rc.canTurn(d))c.rc.turn(d);
            if(c.rc.canTransferCheese(p,c.rc.getRawCheese())){c.rc.transferCheese(p,c.rc.getRawCheese());return;}
        }
    }
    static void face(Context c,MapLocation p) throws GameActionException {if(p==null)return;Direction d=c.rc.getLocation().directionTo(p);if(d!=Direction.CENTER && d!=c.rc.getDirection() && c.rc.canTurn(d))c.rc.turn(d);}

    // mode0: respect cooperation, mode1: enemy king priority, mode2: enemy baby,
    // mode3: cat only. Candidate footprint tiles must lie in the real bite radius.
    static MapLocation attackTarget(Context c,int mode) {
        if(!c.rc.isActionReady())return null;
        int range=c.rc.getType()==UnitType.RAT_KING?GameConstants.RAT_KING_ATTACK_DISTANCE_SQUARED:GameConstants.ATTACK_DISTANCE_SQUARED;
        MapLocation best=null;double score=-Double.MAX_VALUE;
        for(InternalRobot r:c.robots) {
            if(r.getTeam()==c.us || !c.world.getObjectInfo().existsRobot(r.getID()) || r.isBeingThrown() || r.isGrabbedByRobot())continue;
            if(mode==3 && r.getTeam()!=Team.NEUTRAL)continue;
            if(mode==0 && c.world.isCooperation() && r.getTeam()!=Team.NEUTRAL)continue;
            for(MapLocation p:r.getAllPartLocations()) {
                int d=c.rc.getLocation().distanceSquaredTo(p);
                if(d==0 || d>range || !c.on(p) || !c.world.isPassable(p) || (!c.rc.canAttack(p) && !canFace(c,p)))continue;
                double value=(r.getType()==UnitType.RAT_KING?10000:r.getType()==UnitType.BABY_RAT?3000:1000)-r.getHealth()-0.01*d;
                if(mode==2 && r.getType()==UnitType.BABY_RAT)value+=10000;
                if(mode==3 && r.getType()==UnitType.CAT)value+=10000;
                if(r.getHealth()<=GameConstants.RAT_BITE_DAMAGE)value+=5000;
                if(value>score){score=value;best=p;}
            }
        }
        return best;
    }
    static boolean attack(Context c,int mode,boolean powered) throws GameActionException {
        if(!c.alive())return false;
        MapLocation target=attackTarget(c,mode);if(target==null)return false;
        if(!c.rc.canAttack(target))face(c,target);
        int spend=0;
        if(powered) {
            int reserve=4*c.world.getTeamInfo().getNumRatKings(c.us);
            spend=Math.min(64,c.rc.getRawCheese()+Math.max(0,c.rc.getGlobalCheese()-reserve));
            InternalRobot victim=c.world.getRobot(target);
            if(victim!=null && victim.getHealth()<=GameConstants.RAT_BITE_DAMAGE)spend=0;
            else if(victim!=null && victim.getHealth()<GameConstants.RAT_BITE_DAMAGE+8)spend=Math.min(spend,(victim.getHealth()-GameConstants.RAT_BITE_DAMAGE)*(victim.getHealth()-GameConstants.RAT_BITE_DAMAGE));
        }
        if(c.rc.canAttack(target,spend)){c.rc.attack(target,spend);return true;}return false;
    }
    static void step(Context c,Direction direction) throws GameActionException {
        if(!c.alive() || direction==null || direction==Direction.CENTER)return;
        face(c,c.rc.getLocation().add(direction));
        if(c.rc.canMove(direction))c.rc.move(direction);
        else {
            MapLocation dirt=leadingDirt(c,direction);if(dirt!=null)c.rc.removeDirt(dirt);
            if(c.rc.canMove(direction))c.rc.move(direction);
        }
    }

    /** Dijkstra over robot-center cells; king footprints, dirt, congestion and trap risk
     * are included. Occupancy beyond the first move is a soft cost, not a prediction
     * that another robot will actually move. Only the first official legal step executes. */
    static final class Routes {
        final Context c;final int[] cost,first;final int start;
        Routes(Context c,boolean safe) {
            this.c=c;cost=new int[c.width*c.height];first=new int[cost.length];Arrays.fill(cost,INF);Arrays.fill(first,-1);
            int[] travel=new int[cost.length];
            for(int i=0;i<travel.length;i++)travel[i]=tileCost(c,i%c.width,i/c.width,safe);
            MapLocation here=c.rc.getLocation();start=here.y*c.width+here.x;cost[start]=0;
            PriorityQueue<int[]> q=new PriorityQueue<>(Comparator.<int[]>comparingInt(a->a[0]).thenComparingInt(a->a[1]));q.add(new int[]{0,start});
            while(!q.isEmpty()) {
                int[] node=q.poll();int idx=node[1];if(node[0]!=cost[idx])continue;
                int x=idx%c.width,y=idx/c.width;
                for(int i=0;i<8;i++) {
                    int nx=x+DIRS[i].dx,ny=y+DIRS[i].dy;
                    if(nx<0 || ny<0 || nx>=c.width || ny>=c.height)continue;
                    int next=ny*c.width+nx,move=travel[next];
                    if(move>=INF)continue;
                    if(idx==start && !c.rc.canMove(DIRS[i]) && leadingDirt(c,DIRS[i])==null)continue;
                    int d=cost[idx]+move+(i%2==1?4:0);
                    if(d<cost[next]){cost[next]=d;first[next]=idx==start?i:first[idx];q.add(new int[]{d,next});}
                }
            }
        }
        int bestCell(List<MapLocation> goals,int range) {
            int best=-1,value=INF;
            int radius=(int)Math.sqrt(range);
            for(MapLocation goal:goals)for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++) {
                if(dx*dx+dy*dy>range)continue;
                int x=goal.x+dx,y=goal.y+dy;
                if(x<0 || y<0 || x>=c.width || y>=c.height)continue;
                int idx=y*c.width+x;
                if(cost[idx]<value || (cost[idx]==value && cost[idx]<INF && (best<0 || idx<best))){best=idx;value=cost[idx];}
            }
            return best;
        }
        void toward(List<MapLocation> goals,int range) throws GameActionException {
            if(goals.isEmpty())return;int best=bestCell(goals,range);
            if(best>=0 && first[best]>=0)step(c,DIRS[first[best]]);
        }
    }
    static int tileCost(Context c,int x,int y,boolean safe) {
        int radius=c.rc.getType()==UnitType.RAT_KING?1:0,cost=10;
        for(int dy=-radius;dy<=radius;dy++)for(int dx=-radius;dx<=radius;dx++) {
            MapLocation p=new MapLocation(x+dx,y+dy);if(!c.on(p) || c.world.getWall(p))return INF;
            if(c.world.getDirt(p))cost+=30;
            InternalRobot r=c.world.getRobot(p);if(r!=null && r!=c.actor)cost+=r.getTeam()==c.us?60:120;
            if(c.world.hasRatTrap(p,c.us.opponent()))cost+=250;
        }
        if(safe){MapLocation p=new MapLocation(x,y);for(InternalRobot r:c.robots)if(r.getTeam()!=c.us){int d=distance(p,r.getAllPartLocations());if(r.getType()==UnitType.CAT && d<=16)cost+=(17-d)*3;else if(r.getType()!=UnitType.CAT && d<=8)cost+=(9-d)*2;}}
        return cost;
    }
    static void approach(Context c,InternalRobot target,int range,boolean safe) throws GameActionException {if(target!=null && c.rc.isMovementReady())new Routes(c,safe).toward(Arrays.asList(target.getAllPartLocations()),range);}
    static void resource(Context c,boolean mines,boolean safe) throws GameActionException {
        if(!c.rc.isMovementReady())return;
        ArrayList<MapLocation> points=terrain(c,mines);if(points.isEmpty() && !mines){resource(c,true,safe);return;}if(points.isEmpty())return;
        Routes routes=new Routes(c,safe);MapLocation target=null;double best=-Double.MAX_VALUE;
        for(MapLocation p:points) {
            int end=routes.bestCell(List.of(p),mines?2:2);if(end<0)continue;
            double score=(mines?40:Math.min(150,c.world.getCheese(p)))/(double)(10+routes.cost[end]);
            if(mines)for(InternalRobot r:c.robots)if(r!=c.actor && r.getTeam()==c.us && r.getLocation().distanceSquaredTo(p)<=9)score*=0.75;
            if(score>best){best=score;target=p;}
        }
        if(target!=null){routes.toward(List.of(target),2);face(c,target);}
    }
    static void retreat(Context c) throws GameActionException {
        Direction best=Direction.CENTER;double value=safety(c,c.rc.getLocation());
        for(Direction d:DIRS)if(c.rc.canMove(d)){double s=safety(c,c.rc.getLocation().add(d));if(s>value){value=s;best=d;}}
        step(c,best);
        InternalRobot enemy=c.nearest(c.us.opponent(),null);if(enemy!=null)face(c,c.nearestPart(enemy));
    }
    static double safety(Context c,MapLocation p) {
        double score=0;for(InternalRobot r:c.robots)if(r.getTeam()!=c.us){int d=distance(p,r.getAllPartLocations());score-= (r.getType()==UnitType.CAT?30:12)/(double)(1+d);}
        InternalRobot king=c.nearest(c.us,UnitType.RAT_KING);if(king!=null)score-=0.005*distance(p,king.getAllPartLocations());
        if(c.world.hasRatTrap(p,c.us.opponent()))score-=100;return score;
    }
    static MapLocation trapLocation(Context c,boolean catTrap) {
        InternalRobot enemy=catTrap?c.nearest(Team.NEUTRAL,UnitType.CAT):c.nearest(c.us.opponent(),null);
        if(enemy==null)return null;
        MapLocation best=null;double score=-Double.MAX_VALUE;MapLocation here=c.rc.getLocation();
        for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){MapLocation p=here.translate(dx,dy);boolean legal=catTrap?c.rc.canPlaceCatTrap(p):c.rc.canPlaceRatTrap(p);if(!legal)continue;
            int d=distance(p,enemy.getAllPartLocations());double s=-d;
            if(d<=2)s+=40;else if(d<=8)s+=10;
            if(s>score){score=s;best=p;}}
        return best;
    }
    static MapLocation removableTrap(Context c) {
        MapLocation here=c.rc.getLocation();for(int dy=-2;dy<=2;dy++)for(int dx=-2;dx<=2;dx++){MapLocation p=here.translate(dx,dy);if(c.rc.canRemoveRatTrap(p) || c.rc.canRemoveCatTrap(p))return p;}return null;
    }
    static MapLocation nearestDirt(Context c) {
        MapLocation best=null;int score=INF;
        for(int y=0;y<c.height;y++)for(int x=0;x<c.width;x++){MapLocation p=new MapLocation(x,y);int d=c.rc.getLocation().distanceSquaredTo(p);if(c.world.getDirt(p) && d<score){score=d;best=p;}}
        return best;
    }
    static InternalRobot rescueTarget(Context c) {
        InternalRobot best=null;int score=INF;
        for(InternalRobot r:c.robots)if(r!=c.actor && r.getTeam()==c.us && r.getType()==UnitType.BABY_RAT && !r.isBeingThrown() && !r.isGrabbedByRobot() && r.getHealth()<50){int d=c.rc.getLocation().distanceSquaredTo(r.getLocation());if(d<score){score=d;best=r;}}
        return best;
    }
    static boolean carry(Context c,InternalRobot r) throws GameActionException {
        if(r==null || c.actor.isCarryingRobot())return false;
        face(c,r.getLocation());if(c.rc.canCarryRat(r.getLocation())){c.rc.carryRat(r.getLocation());return true;}return false;
    }
    static Direction throwDirection(Context c,boolean transport) {
        if(!c.actor.isCarryingRobot())return null;
        InternalRobot held=c.actor.getRobotBeingCarried(),goal=c.nearest(c.us.opponent(),UnitType.RAT_KING);
        Direction best=null;double score=-Double.MAX_VALUE;
        for(Direction d:DIRS) {
            if(!c.rc.canDropRat(d) || (d!=c.rc.getDirection() && !c.rc.canTurn(d)))continue;
            MapLocation p=c.rc.getLocation();double s=0;
            for(int n=1;n<=8;n++) {
                p=p.add(d);
                if(!c.on(p) || !c.world.isPassable(p)){s+=(held.getTeam()!=c.us?100: -100)-n*3;break;}
                InternalRobot hit=c.world.getRobot(p);
                if(hit!=null && hit!=c.actor){s+=hit.getTeam()==c.us?-200:hit.getType()==UnitType.RAT_KING?300:hit.getType()==UnitType.CAT?-50:150;break;}
                if(transport && goal!=null)s=100-distance(p,goal.getAllPartLocations());
                else if(held.getTeam()!=c.us)s=30-n;
            }
            if(s>score){score=s;best=d;}
        }
        return best;
    }
    static void throwRat(Context c,boolean transport) throws GameActionException {Direction d=throwDirection(c,transport);if(d==null)return;face(c,c.rc.getLocation().add(d));if(c.rc.canThrowRat())c.rc.throwRat();}
    static void drop(Context c) throws GameActionException {
        Direction best=null;double score=-Double.MAX_VALUE;for(Direction d:DIRS)if(c.rc.canDropRat(d)){double s=safety(c,c.rc.getLocation().add(d));if(s>score){score=s;best=d;}}if(best!=null)c.rc.dropRat(best);
    }
    static void buildToward(Context c,MapLocation goal) throws GameActionException {
        if(goal==null)goal=c.rc.getLocation().add(Direction.NORTH);
        MapLocation best=null;int score=INF;
        for(Direction d:DIRS){MapLocation p=c.session.buildLocation(c.rc,d);if(p!=null){int s=p.distanceSquaredTo(goal);if(s<score){score=s;best=p;}}}
        if(best!=null){face(c,best);if(c.rc.canBuildRat(best))c.rc.buildRat(best);}
    }
    static void formKing(Context c) throws GameActionException {
        if(c.rc.canBecomeRatKing()){c.rc.becomeRatKing();return;}
        if(c.rc.getType()!=UnitType.BABY_RAT || !c.rc.isMovementReady())return;
        MapLocation best=null;double score=-Double.MAX_VALUE;
        for(InternalRobot candidate:c.robots)if(candidate.getTeam()==c.us && candidate.getType()==UnitType.BABY_RAT && !candidate.isBeingThrown() && !candidate.isGrabbedByRobot()) {
            MapLocation center=candidate.getLocation();boolean clear=true;int neighbors=0;
            for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){MapLocation p=center.translate(dx,dy);if(!c.on(p)||!c.world.isPassable(p)){clear=false;continue;}InternalRobot r=c.world.getRobot(p);if(r!=null){if(r.getType()!=UnitType.BABY_RAT)clear=false;else if(r.getTeam()==c.us)neighbors++;}}
            double s=neighbors*50-c.rc.getLocation().distanceSquaredTo(center);
            if(clear && s>score){score=s;best=center;}
        }
        if(best!=null){ArrayList<MapLocation> goals=new ArrayList<>();for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){MapLocation p=best.translate(dx,dy);if(c.world.getRobot(p)==null || c.world.getRobot(p)==c.actor)goals.add(p);}new Routes(c,true).toward(goals,0);}
        if(c.rc.canBecomeRatKing())c.rc.becomeRatKing();
    }
    static void economy(Context c,boolean peaceful) throws GameActionException {
        InternalRobot king=c.nearest(c.us,UnitType.RAT_KING);
        if(c.rc.getType()==UnitType.BABY_RAT && c.rc.getRawCheese()>=10 && king!=null){deposit(c);if(c.rc.getRawCheese()>0)approach(c,king,GameConstants.CHEESE_TRANSFER_RADIUS_SQUARED,true);deposit(c);}
        else resource(c,false,true);
        if(!peaceful)attack(c,0,false);else attack(c,3,false);
    }

    static void execute(EngineBridge.Session session,int action) {
        Context c=new Context(session);
        if(c.actor.isGrabbedByRobot() || c.actor.isBeingThrown())return;
        try {
            pickup(c);
            boolean combat=(action>=1 && action<=8) || action==21 || action==22 || action==32 || action==37;
            int attackMode=action==23?3:action==22?2:combat?1:0;
            // Resource actions reserve their action cooldown for transfer/build/trap.
            if(combat && action!=32)attack(c,attackMode,action==37);
            if(action>=1 && action<=8)step(c,DIRS[action-1]);
            else if(action>=9 && action<=16){MapLocation p=session.buildLocation(c.rc,DIRS[action-9]);if(p!=null){face(c,p);if(c.rc.canBuildRat(p))c.rc.buildRat(p);}}
            else switch(action) {
                case 0: attack(c,0,false);break;
                case 17: economy(c,false);break;
                case 18:{deposit(c);InternalRobot king=c.nearest(c.us,UnitType.RAT_KING);if(c.rc.getRawCheese()>0)approach(c,king,GameConstants.CHEESE_TRANSFER_RADIUS_SQUARED,true);deposit(c);break;}
                case 19: resource(c,true,true);break;
                case 20:{InternalRobot king=c.nearest(c.us,UnitType.RAT_KING);InternalRobot threat=null;int d=INF;for(InternalRobot r:c.robots)if(r.getTeam()!=c.us && king!=null){int n=distance(r.getLocation(),king.getAllPartLocations());if(n<d){d=n;threat=r;}}if(threat!=null && d<36){attack(c,1,false);approach(c,threat,2,false);}else approach(c,king,9,true);break;}
                case 21: approach(c,c.nearest(c.us.opponent(),UnitType.RAT_KING),c.rc.getType()==UnitType.RAT_KING?GameConstants.RAT_KING_ATTACK_DISTANCE_SQUARED:2,false);break;
                case 22: approach(c,c.nearest(c.us.opponent(),UnitType.BABY_RAT),2,false);break;
                case 23: attack(c,3,false);approach(c,c.nearest(Team.NEUTRAL,UnitType.CAT),2,false);break;
                case 24: retreat(c);break;
                case 25: resource(c,false,true);break;
                case 26:{MapLocation goal=nearestPoint(c.rc.getLocation(),terrain(c,false));if(goal==null)goal=nearestPoint(c.rc.getLocation(),terrain(c,true));buildToward(c,goal);break;}
                case 27: buildToward(c,c.nearestPart(c.nearest(c.us.opponent(),UnitType.RAT_KING)));break;
                case 28:{MapLocation p=trapLocation(c,false);if(p!=null)c.rc.placeRatTrap(p);break;}
                case 29:{MapLocation p=trapLocation(c,true);if(p!=null)c.rc.placeCatTrap(p);break;}
                case 30:{MapLocation p=removableTrap(c);if(p!=null){if(c.rc.canRemoveRatTrap(p))c.rc.removeRatTrap(p);else if(c.rc.canRemoveCatTrap(p))c.rc.removeCatTrap(p);}break;}
                case 31: formKing(c);break;
                case 32:{InternalRobot r=c.nearest(c.us.opponent(),UnitType.BABY_RAT);if(!carry(c,r)){approach(c,r,2,false);carry(c,r);}break;}
                case 33:{InternalRobot r=rescueTarget(c);if(!carry(c,r)){approach(c,r,2,true);carry(c,r);}break;}
                case 34: throwRat(c,false);break;
                case 35: throwRat(c,true);break;
                case 36: drop(c);break;
                case 37: attack(c,1,true);break;
                case 38:{MapLocation p=nearestDirt(c);if(p!=null){face(c,p);if(c.rc.canRemoveDirt(p))c.rc.removeDirt(p);else if(c.rc.isMovementReady())new Routes(c,false).toward(List.of(p),2);}break;}
                case 39: economy(c,true);break;
                default:throw new IllegalArgumentException("macro_v2 action out of range: "+action);
            }
            if(!c.alive())return;
            pickup(c);
            if(combat || action==23)attack(c,attackMode,action==37);
            if(!combat && action!=32 && action!=33 && action!=34 && action!=35)deposit(c);
        } catch(GameActionException e) {throw new IllegalStateException("macro_v2 controller action failed",e);}
    }
}
