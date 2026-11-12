package alpha0.imitation;
import battlecode.common.*;
import java.lang.reflect.*;
import java.nio.file.*;
import java.io.*;
import java.util.*;

/** Independent public-interface observer; original controller and sandbox remain unmodified. */
public final class ControllerCallAudit {
 private static final Set<String> NAMES=Set.of("placeDirt","placeRatTrap","removeRatTrap","placeCatTrap","removeCatTrap","removeDirt","pickUpCheese","moveForward","move","turn","buildRat","attack","becomeRatKing","squeak","writeSharedArray","transferCheese","throwRat","dropRat","carryRat","disintegrate","resign");
 private final List<Object> rows=new ArrayList<>();private long bytes;private boolean failed;
 private Object value(Object v){if(v instanceof MapLocation l)return List.of(l.x,l.y);if(v instanceof Direction d)return d.name();if(v==null||v instanceof Integer)return v;throw new IllegalArgumentException("unsupported argument");}
 public RobotController wrap(RobotController original){
  return (RobotController)Proxy.newProxyInstance(RobotController.class.getClassLoader(),new Class<?>[]{RobotController.class},(proxy,method,args)->{
   if(!NAMES.contains(method.getName())){try{return method.invoke(original,args);}catch(InvocationTargetException e){throw e.getCause();}}
   List<Object> values=new ArrayList<>();if(args!=null)for(Object a:args)values.add(value(a));
   String op=method.getName();if((op.equals("attack")||op.equals("pickUpCheese"))&&values.size()==2)op+="Amount";
   Map<String,Object> row=new TreeMap<>();row.put("robot",original.getID());row.put("team",original.getTeam().name());row.put("round",original.getRoundNum());row.put("operation",op);row.put("arguments",values);
   Object result=null;Throwable failure=null;
   try{result=method.invoke(original,args);return result;}
   catch(InvocationTargetException e){failure=e.getCause();throw failure;}
   finally{row.put("return_boolean",result instanceof Boolean?result:null);row.put("exception",failure==null?null:failure.getClass().getName());append(row);}
  });
 }
 private synchronized void append(Map<String,Object> row){
  if(failed)throw new IllegalStateException("call audit failed");
  try{row.put("sequence",rows.size());long n=TraceJournal.canonical(row).length;if(n>1024*1024||n>8L*1024*1024-bytes)throw new IllegalStateException("call audit cap");rows.add(row);bytes+=n;}
  catch(IOException|RuntimeException e){failed=true;throw new IllegalStateException(e);}
 }
 public void save(Path path)throws IOException{if(failed)throw new IllegalStateException("call audit failed");Files.write(path,TraceJournal.canonical(rows),StandardOpenOption.CREATE_NEW);}
}
