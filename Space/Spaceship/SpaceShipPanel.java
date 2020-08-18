import sun.audio.*;
import java.io.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;

/**
 *SpaceShipPanel.java
 *
 *The SpaceShipPanel creates the ship and all it's componenets. 
 */
public class SpaceShipPanel extends JPanel
{
  private final int radius1 = 4, radius2 = 10;
  private int numClick = 0;
  private Point point1 = null, point2 = null;
  private JButton reset;

  /**
   * The SpaceShipPanel method creates a panel and listeners for buttons and shots
   * 
   * @param None
   * @return None
   */
  public SpaceShipPanel()
  {
    reset = new JButton("Reset");
    reset.addActionListener(new ButtonListener());
    add(reset);

    ShotsListener listener = new ShotsListener();
    addMouseListener(listener);
    addMouseMotionListener(listener);

    setBackground(Color.orange);
    setPreferredSize(new Dimension(300, 200));
  }

  /**
   * The paintComponent method creates a ship and draws the lasers
   * 
   * @param None
   * @return None
   */
  public void paintComponent (Graphics page)
  {
    super.paintComponent(page);

    page.setColor(Color.black);
    page.drawString("USS Enterprise", 0, 60);
    page.drawString("Shot#" + numClick, 100, 60);

    if(point1 != null)
    {
      page.setColor(Color.cyan);
      page.fillOval(point1.x - radius1, point1.y - radius1, radius1 * 2, radius1 * 2);
      page.setColor(Color.blue);
      page.fillOval(point1.x - radius2, point1.y, radius2 * 2, 8);
    }
    if(point2 != null)
    {
       int caseNum = numClick % 4;

      switch (caseNum)
      {
        case 0:
          page.setColor(Color.red);
          break;
        case 2:
          page.setColor(Color.black);
          break;
        case 4:
	  page.setColor(Color.blue);
          break;
        case 3:
          page.setColor(Color.green);
          break;
       }

      page.drawLine(point1.x, point1.y + radius2, (int)(Math.random() * 300)+100, (int)(Math.random() * 200)+100);
      point2 = null;
    }
  }

  /**
   * The ShotsListener implements the listener events for sounds
   */
  private class ShotsListener implements MouseListener, MouseMotionListener
  {
    /**
     * The mousePressed method implements actions when the mouse is pressed. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mousePressed(MouseEvent event){}

    /**
     * The mouseMoved method implements actions when the mouse is moved. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseMoved(MouseEvent event)
    {
      point1 = event.getPoint();
      repaint();
    }

    /**
     * The mouseClicked method implements sounds and draws lines when the mouse is clicked. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseClicked(MouseEvent event)
    {
      numClick++;
      point2 = event.getPoint();
      try
      {

      InputStream in = new FileInputStream("bonk.au");
      AudioStream as = new AudioStream(in);
      AudioPlayer.player.start(as);
      }
       catch (Exception e)
      {
      } 
      repaint();
    }

    /**
     * The mouseDragged method implements actions when the mouse is dragged 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseDragged(MouseEvent event) {}

    /**
     * The mouseReleased method implements actions when the mouse is released 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseReleased(MouseEvent event) {}

    /**
     * The mouseEntered method implements actions when the mouse is entered. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseEntered(MouseEvent event) {}

    /**
     * The mouseExited method implements actions when the mouse is exited. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseExited(MouseEvent event) {}
  }

  /**
   * The DotsListener class implements the listener events for lasers
   */ 
  private class DotsListener implements MouseListener, MouseMotionListener
  {

    /**
     * The mousePressed method implements actions when the mouse is pressed. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mousePressed(MouseEvent event)
    {
      repaint();
    }

    /**
     * The mouseMoved method implements actions when the mouse is moved. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseMoved(MouseEvent event)
    {
      point1 = event.getPoint();
      repaint();
    }

    /**
     * The mouseClicked method implements actions when the mouse is clicked. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseClicked(MouseEvent event)
    {
      numClick++;
      point2 = event.getPoint();
      repaint();
    }

    /**
     * The mouseDragged method implements actions when the mouse is dragged. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseDragged(MouseEvent event) {}

    /**
     * The mouseReleased method implements actions when the mouse is released. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseReleased(MouseEvent event) {}

    /**
     * The mouseEntered method implements actions when the mouse is entered. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseEntered(MouseEvent event) {}

    /**
     * The mouseExited method implements actions when the mouse is exited. 
     * 
     * @param event The mouse action
     * @return None
     */
    public void mouseExited(MouseEvent event) {}
  }

  /**
   * The ButtonListener implements the listener events for the button
   */
  private class ButtonListener implements ActionListener
  {

    /**
     * The actionPerformed method implements actions when the button is pressed. 
     * 
     * @param event The button was pressed
     * @return None
     */
    public void actionPerformed(ActionEvent event)
    {
      numClick = 0;
    }
  }
}
