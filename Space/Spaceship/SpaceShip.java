import javax.swing.JFrame;

/**
 *SpaceShip.java
 *
 *The SpaceShip program has a ship follow your mouse and fires lasers when clicked. 
 */

/**
 * Algorithm
 * 
 * 1) Draw the background
 * 2) Create a counter and a reset button
 * 3) Draw the ship
 * 4) Have the sip fire lasers when clicked
 * 5) Have sounds play when lasers are shot. 
 * 
 */

public class SpaceShip
{
  public static void main(String[] args)
  {
    JFrame frame = new JFrame ("SpaceShip");
    frame.setDefaultCloseOperation (JFrame.EXIT_ON_CLOSE);

    frame.getContentPane().add (new SpaceShipPanel());

    frame.pack();
    frame.setVisible(true);
  }
}
